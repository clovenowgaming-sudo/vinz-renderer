#include "Renderer.hpp"
#include "gl/GL.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace geode::prelude;

// ======================================================================
//  Vinz Renderer - desain "static quad bake"
//
//  1. Saat level selesai dimuat, object statis (dekorasi / solid tanpa group,
//     tanpa HSV, tanpa animasi) dibaca sprite-nya dan geometrinya (4 vertex
//     per sprite) dihitung sekali lalu dikirim ke VBO GPU.
//  2. Warna object TIDAK ditulis ke vertex. Vertex hanya menyimpan nomor
//     color channel; warna diambil di vertex shader dari tekstur 1024x1
//     yang di-update sekali per frame dari GJEffectManager.
//     => color/pulse trigger jalan tanpa menyentuh object satu-satu.
//  3. Quad diurutkan per chunk (sumbu X) lalu per urutan atlas asli, jadi
//     chunk yang kelihatan selalu satu rentang index yang menyambung.
//       - ES 3.0 / desktop : CPU cek chunk vs layar, glDrawElements.
//       - ES 3.1           : compute shader cek chunk, tulis perintah
//                            glDrawElementsIndirect langsung di GPU.
//  4. Sprite asli disembunyikan (hook GameObject::setVisible di main.cpp).
// ======================================================================

namespace vr {
namespace {

struct Vertex {
    float x, y, u, v;
    uint8_t r, g, b, a;
    uint16_t channel;
    uint16_t pad;
};
static_assert(sizeof(Vertex) == 24, "Vertex harus 24 byte");

struct Quad {
    Vertex v[4];
    int atlas = 0;
    float cx = 0.f;
    int chunk = 0;
};

// Layout cocok dengan std430 di compute shader (32 byte).
struct Chunk {
    float minX, minY, maxX, maxY;
    uint32_t first, count, pad0, pad1;
};
static_assert(sizeof(Chunk) == 32, "Chunk harus 32 byte");

struct Config {
    bool enabled = true;
    bool deco = true;
    bool solids = true;
    bool debug = false;
    int chunkSize = 256;
    int marginPct = 5;
    std::string mode = "Auto";
};

constexpr int kChannelCount = 1024;
constexpr int kBlackChannel = 1010;

std::unordered_set<GameObject*> g_baked;

// ---------------- GL objects bersama ----------------
GLuint s_prog = 0;
GLuint s_cullProg = 0;
GLuint s_chanTex = 0;
GLint uMvp = -1, uTex = -1, uChan = -1, uPremul = -1;
GLint cMvp = -1, cCount = -1, cMargin = -1;
uint8_t s_chanData[kChannelCount * 4];
unsigned int s_chanFrame = 0xffffffff;

int s_frame = -1;
int s_drawnAcc = 0;
int s_drawnLast = 0;

char const* kVertSrc = R"(
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;
layout(location = 3) in uint a_channel;
uniform mat4 u_mvp;
uniform sampler2D u_channels;
out vec2 v_uv;
out vec4 v_color;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
    v_color = (a_channel == 0u)
        ? a_color
        : texelFetch(u_channels, ivec2(int(a_channel), 0), 0);
}
)";

char const* kFragSrc = R"(
in vec2 v_uv;
in vec4 v_color;
uniform sampler2D u_tex;
uniform int u_premul;
layout(location = 0) out vec4 o_color;
void main() {
    vec4 vc = v_color;
    if (u_premul != 0) vc = vec4(vc.rgb * vc.a, vc.a);
    vec4 c = texture(u_tex, v_uv) * vc;
    if (c.a < 0.004) discard;
    o_color = c;
}
)";

char const* kCullSrc = R"(
layout(local_size_x = 64) in;
struct Chunk { vec4 bounds; uint first; uint count; uint pad0; uint pad1; };
layout(std430, binding = 0) readonly buffer Chunks { Chunk chunks[]; };
layout(std430, binding = 1) buffer Cmd {
    uint count;
    uint instanceCount;
    uint firstIndex;
    int  baseVertex;
    uint baseInstance;
} cmd;
uniform mat4 u_mvp;
uniform int u_chunkCount;
uniform float u_margin;
shared int sFirst;
shared int sLast;

bool chunkVisible(vec4 b) {
    vec2 lo = vec2( 1e30);
    vec2 hi = vec2(-1e30);
    vec2 pts[4] = vec2[4](b.xy, vec2(b.z, b.y), vec2(b.x, b.w), b.zw);
    for (int k = 0; k < 4; k++) {
        vec4 c = u_mvp * vec4(pts[k], 0.0, 1.0);
        if (c.w <= 0.0) return true;
        vec2 n = c.xy / c.w;
        lo = min(lo, n);
        hi = max(hi, n);
    }
    float m = 1.0 + u_margin;
    return hi.x >= -m && lo.x <= m && hi.y >= -m && lo.y <= m;
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    if (tid == 0u) { sFirst = 2147483647; sLast = -1; }
    memoryBarrierShared();
    barrier();

    for (int i = int(tid); i < u_chunkCount; i += 64) {
        if (chunkVisible(chunks[i].bounds)) {
            atomicMin(sFirst, i);
            atomicMax(sLast, i);
        }
    }
    memoryBarrierShared();
    barrier();

    if (tid == 0u) {
        cmd.instanceCount = 1u;
        cmd.baseVertex = 0;
        cmd.baseInstance = 0u;
        if (sLast < 0) {
            cmd.count = 0u;
            cmd.firstIndex = 0u;
        } else {
            uint f = chunks[sFirst].first;
            uint l = chunks[sLast].first + chunks[sLast].count;
            cmd.count = l - f;
            cmd.firstIndex = f;
        }
    }
}
)";

bool buildPrograms(bool wantCompute) {
    if (!s_prog) {
        auto hdr = vgl::glslHeader(false);
        auto vs = vgl::compileShader(GL_VERTEX_SHADER, hdr + kVertSrc, "vr.vert");
        auto fs = vgl::compileShader(GL_FRAGMENT_SHADER, hdr + kFragSrc, "vr.frag");
        if (!vs || !fs) return false;
        s_prog = vgl::linkProgram(vs, fs);
        if (!s_prog) return false;
        uMvp = glGetUniformLocation(s_prog, "u_mvp");
        uTex = glGetUniformLocation(s_prog, "u_tex");
        uChan = glGetUniformLocation(s_prog, "u_channels");
        uPremul = glGetUniformLocation(s_prog, "u_premul");
    }
    if (wantCompute && !s_cullProg) {
        auto cs = vgl::compileShader(GL_COMPUTE_SHADER, vgl::glslHeader(true) + kCullSrc, "vr.cull.comp");
        s_cullProg = vgl::linkProgram(cs);
        if (s_cullProg) {
            cMvp = glGetUniformLocation(s_cullProg, "u_mvp");
            cCount = glGetUniformLocation(s_cullProg, "u_chunkCount");
            cMargin = glGetUniformLocation(s_cullProg, "u_margin");
        } else {
            log::warn("Compute culling gagal dibuat, fallback ke CPU culling");
        }
    }
    return true;
}

void ensureChannelTexture() {
    if (s_chanTex) return;
    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &s_chanTex);
    glBindTexture(GL_TEXTURE_2D, s_chanTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    std::memset(s_chanData, 255, sizeof(s_chanData));
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, kChannelCount, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, s_chanData);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
}

Config readConfig() {
    auto mod = Mod::get();
    Config c;
    c.enabled = mod->getSettingValue<bool>("enabled");
    c.deco = mod->getSettingValue<bool>("bake-decoration");
    c.solids = mod->getSettingValue<bool>("bake-solids");
    c.debug = mod->getSettingValue<bool>("debug-overlay");
    c.chunkSize = (int)mod->getSettingValue<int64_t>("chunk-size");
    c.marginPct = (int)mod->getSettingValue<int64_t>("cull-margin");
    c.mode = mod->getSettingValue<std::string>("renderer-mode");
    return c;
}

// ---------------- Bake sprite -> quad ----------------

bool bakeSprite(CCSprite* spr, CCAffineTransform const& toBatch, int channel, Quad& out) {
    auto tex = spr->getTexture();
    if (!tex) return false;
    float aw = (float)tex->getPixelsWide();
    float ah = (float)tex->getPixelsHigh();
    if (aw <= 0.f || ah <= 0.f) return false;

    CCRect r = spr->getTextureRect();
    float s = CC_CONTENT_SCALE_FACTOR();
    float rx = r.origin.x * s, ry = r.origin.y * s;
    float rw = r.size.width * s, rh = r.size.height * s;
    if (rw <= 0.f || rh <= 0.f) return false;

    float left, right, top, bottom;
    float uv[4][2]; // bl, br, tl, tr
    if (!spr->isTextureRectRotated()) {
        left = rx / aw;
        right = (rx + rw) / aw;
        top = ry / ah;
        bottom = (ry + rh) / ah;
        if (spr->isFlipX()) std::swap(left, right);
        if (spr->isFlipY()) std::swap(top, bottom);
        uv[0][0] = left;  uv[0][1] = bottom;
        uv[1][0] = right; uv[1][1] = bottom;
        uv[2][0] = left;  uv[2][1] = top;
        uv[3][0] = right; uv[3][1] = top;
    } else {
        left = rx / aw;
        right = (rx + rh) / aw;
        top = ry / ah;
        bottom = (ry + rw) / ah;
        if (spr->isFlipX()) std::swap(top, bottom);
        if (spr->isFlipY()) std::swap(left, right);
        uv[0][0] = left;  uv[0][1] = top;
        uv[1][0] = left;  uv[1][1] = bottom;
        uv[2][0] = right; uv[2][1] = top;
        uv[3][0] = right; uv[3][1] = bottom;
    }

    CCPoint off = spr->getOffsetPosition();
    float x1 = off.x, y1 = off.y;
    float x2 = x1 + r.size.width, y2 = y1 + r.size.height;
    CCPoint p[4] = {
        CCPointApplyAffineTransform(CCPoint(x1, y1), toBatch),
        CCPointApplyAffineTransform(CCPoint(x2, y1), toBatch),
        CCPointApplyAffineTransform(CCPoint(x1, y2), toBatch),
        CCPointApplyAffineTransform(CCPoint(x2, y2), toBatch),
    };

    ccColor3B col = spr->getColor();
    uint8_t opa = spr->getOpacity();
    float sumX = 0.f;
    for (int i = 0; i < 4; i++) {
        Vertex& v = out.v[i];
        v.x = (float)p[i].x;
        v.y = (float)p[i].y;
        v.u = uv[i][0];
        v.v = uv[i][1];
        if (channel > 0) {
            v.r = v.g = v.b = v.a = 255; // warna diambil dari channel
        } else {
            v.r = col.r; v.g = col.g; v.b = col.b; v.a = opa;
        }
        v.channel = (uint16_t)std::max(channel, 0);
        v.pad = 0;
        sumX += v.x;
    }
    out.atlas = (int)spr->getAtlasIndex();
    out.cx = sumX * 0.25f;
    return true;
}

bool isEligible(GameObject* o, Config const& c) {
    if (!o) return false;
    if (o->isTrigger() || o->m_isHide) return false;
    if (o->m_groupCount != 0) return false;

    bool typeOk =
        (c.deco && o->m_objectType == GameObjectType::Decoration) ||
        (c.solids && o->m_objectType == GameObjectType::Solid);
    if (!typeOk) return false;

    if (o->m_baseColor && o->m_baseColor->m_usesHSV) return false;
    if (o->m_detailColor && o->m_detailColor->m_usesHSV) return false;
    if (o->numberOfRunningActions() > 0) return false;

    if (!typeinfo_cast<CCSpriteBatchNode*>(o->getParent())) return false;

    int mainCh = o->m_activeMainColorID;
    int detCh = o->m_activeDetailColorID;
    if (mainCh < 0 || mainCh > kBlackChannel) return false;
    if (detCh < 0 || detCh > kBlackChannel) return false;

    if (auto kids = o->getChildren()) {
        for (auto kid : CCArrayExt<CCNode*>(kids)) {
            auto spr = typeinfo_cast<CCSprite*>(kid);
            if (!spr) return false;
            if (spr->getChildrenCount() > 0) return false;
            if (spr->numberOfRunningActions() > 0) return false;
        }
    }
    return true;
}

bool bakeObject(GameObject* o, std::vector<Quad>& out) {
    int mainCh = o->m_activeMainColorID;
    int detCh = o->m_activeDetailColorID;

    CCAffineTransform base = o->nodeToParentTransform();
    Quad q;
    if (bakeSprite(o, base, mainCh, q)) out.push_back(q);

    if (auto kids = o->getChildren()) {
        for (auto kid : CCArrayExt<CCNode*>(kids)) {
            auto spr = static_cast<CCSprite*>(kid);
            if (!spr->isVisible()) continue;
            CCAffineTransform t = CCAffineTransformConcat(spr->nodeToParentTransform(), base);
            Quad cq;
            if (bakeSprite(spr, t, detCh > 0 ? detCh : mainCh, cq)) out.push_back(cq);
        }
    }
    return true;
}

// ---------------- Node renderer ----------------

class BatchNode : public CCNode {
public:
    static BatchNode* create(
        CCSpriteBatchNode* batch, std::vector<Quad>& quads, std::vector<GameObject*> objs,
        Config const& cfg, PlayLayer* layer, bool gpuCull
    ) {
        auto ret = new BatchNode();
        if (ret && ret->setup(batch, quads, std::move(objs), cfg, layer, gpuCull)) {
            ret->autorelease();
            return ret;
        }
        delete ret;
        return nullptr;
    }

    ~BatchNode() override {
        for (auto o : m_objs) g_baked.erase(o);
        if (m_vbo) glDeleteBuffers(1, &m_vbo);
        if (m_ibo) glDeleteBuffers(1, &m_ibo);
        if (m_ssbo) glDeleteBuffers(1, &m_ssbo);
        if (m_indirect) glDeleteBuffers(1, &m_indirect);
        if (m_vao) vgl::deleteVertexArrays(1, &m_vao);
    }

    void setLabel(CCLabelBMFont* label) { m_label = label; }
    size_t quadCount() const { return m_quadCount; }

    void draw() override {
        if (!m_batch || !m_batch->isVisible()) return;
        if (!m_uploaded && !upload()) { m_failed = true; }
        if (m_failed) return;

        updateChannels();

        kmMat4 mv, pr, mvp;
        kmGLGetMatrix(KM_GL_MODELVIEW, &mv);
        kmGLGetMatrix(KM_GL_PROJECTION, &pr);
        kmMat4Multiply(&mvp, &pr, &mv);

        float margin = m_marginPct / 100.f * 2.f;

        // --- bagian culling ---
        bool useGpu = m_gpuCull && s_cullProg && m_ssbo && m_indirect;
        std::vector<std::pair<uint32_t, uint32_t>> runs; // (firstIndex, count) untuk jalur CPU

        if (useGpu) {
            glUseProgram(s_cullProg);
            glUniformMatrix4fv(cMvp, 1, GL_FALSE, mvp.mat);
            glUniform1i(cCount, (GLint)m_chunks.size());
            glUniform1f(cMargin, margin);
            vgl::bindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssbo);
            vgl::bindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_indirect);
            vgl::dispatchCompute(1, 1, 1);
            vgl::memoryBarrier(GL_COMMAND_BARRIER_BIT);
        } else {
            cullCpu(mvp.mat, margin, runs);
            if (runs.empty()) { finishFrameStats(0); return; }
        }

        // --- bagian menggambar ---
        glUseProgram(s_prog);
        glUniformMatrix4fv(uMvp, 1, GL_FALSE, mvp.mat);
        glUniform1i(uTex, 0);
        glUniform1i(uChan, 1);
        glUniform1i(uPremul, m_blend.src == GL_ONE ? 1 : 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, s_chanTex);
        glActiveTexture(GL_TEXTURE0);
        ccGLBindTexture2D(m_texName);
        ccGLBlendFunc(m_blend.src, m_blend.dst);

        vgl::bindVertexArray(m_vao);
        int drawnQuads = 0;
        if (useGpu) {
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_indirect);
            vgl::drawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
            drawnQuads = -1; // tidak diketahui tanpa readback
        } else {
            for (auto& run : runs) {
                glDrawElements(
                    GL_TRIANGLES, (GLsizei)run.second, GL_UNSIGNED_INT,
                    (void const*)(uintptr_t)(run.first * sizeof(uint32_t))
                );
                drawnQuads += (int)(run.second / 6);
            }
        }
        vgl::bindVertexArray(0);

        // --- kembalikan state biar cocos tidak bingung ---
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
        glUseProgram(0);
        ccGLInvalidateStateCache();

        finishFrameStats(drawnQuads);
    }

private:
    CCSpriteBatchNode* m_batch = nullptr;
    PlayLayer* m_layer = nullptr;
    CCLabelBMFont* m_label = nullptr;
    GLuint m_texName = 0;
    ccBlendFunc m_blend{GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA};
    std::vector<GameObject*> m_objs;
    std::vector<Vertex> m_verts;
    std::vector<uint32_t> m_indices;
    std::vector<Chunk> m_chunks;
    size_t m_quadCount = 0;
    int m_marginPct = 5;
    bool m_gpuCull = false;
    bool m_uploaded = false;
    bool m_failed = false;
    bool m_debug = false;
    GLuint m_vao = 0, m_vbo = 0, m_ibo = 0, m_ssbo = 0, m_indirect = 0;

    bool setup(
        CCSpriteBatchNode* batch, std::vector<Quad>& quads, std::vector<GameObject*> objs,
        Config const& cfg, PlayLayer* layer, bool gpuCull
    ) {
        if (!CCNode::init()) return false;
        m_batch = batch;
        m_layer = layer;
        m_objs = std::move(objs);
        m_marginPct = cfg.marginPct;
        m_gpuCull = gpuCull;
        m_debug = cfg.debug;
        m_blend = batch->getBlendFunc();
        if (auto tex = batch->getTexture()) m_texName = tex->getName();
        if (!m_texName) return false;

        float chunkSize = (float)std::max(cfg.chunkSize, 16);
        for (auto& q : quads) q.chunk = (int)std::floor(q.cx / chunkSize);
        std::stable_sort(quads.begin(), quads.end(), [](Quad const& a, Quad const& b) {
            if (a.chunk != b.chunk) return a.chunk < b.chunk;
            return a.atlas < b.atlas;
        });

        m_quadCount = quads.size();
        m_verts.reserve(quads.size() * 4);
        m_indices.reserve(quads.size() * 6);

        int curChunk = std::numeric_limits<int>::min();
        float inf = std::numeric_limits<float>::infinity();
        for (auto& q : quads) {
            if (q.chunk != curChunk || m_chunks.empty()) {
                curChunk = q.chunk;
                Chunk c{};
                c.minX = inf; c.minY = inf; c.maxX = -inf; c.maxY = -inf;
                c.first = (uint32_t)m_indices.size();
                c.count = 0;
                m_chunks.push_back(c);
            }
            Chunk& c = m_chunks.back();
            uint32_t base = (uint32_t)m_verts.size();
            for (int i = 0; i < 4; i++) {
                m_verts.push_back(q.v[i]);
                c.minX = std::min(c.minX, q.v[i].x);
                c.minY = std::min(c.minY, q.v[i].y);
                c.maxX = std::max(c.maxX, q.v[i].x);
                c.maxY = std::max(c.maxY, q.v[i].y);
            }
            // urutan vertex: bl, br, tl, tr
            uint32_t idx[6] = {base, base + 1, base + 2, base + 3, base + 2, base + 1};
            m_indices.insert(m_indices.end(), idx, idx + 6);
            c.count += 6;
        }
        return !m_chunks.empty();
    }

    bool upload() {
        glGenBuffers(1, &m_vbo);
        glGenBuffers(1, &m_ibo);
        vgl::genVertexArrays(1, &m_vao);
        vgl::bindVertexArray(m_vao);

        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(m_verts.size() * sizeof(Vertex)), m_verts.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(m_indices.size() * sizeof(uint32_t)), m_indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void const*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void const*)8);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), (void const*)16);
        glEnableVertexAttribArray(3);
        vgl::vertexAttribIPointer(3, 1, GL_UNSIGNED_SHORT, sizeof(Vertex), (void const*)20);

        vgl::bindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        if (m_gpuCull && s_cullProg) {
            glGenBuffers(1, &m_ssbo);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo);
            glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(m_chunks.size() * sizeof(Chunk)), m_chunks.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

            uint32_t zero[5] = {0, 1, 0, 0, 0};
            glGenBuffers(1, &m_indirect);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_indirect);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero), zero, GL_DYNAMIC_COPY);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        // CPU copy tidak dibutuhkan lagi
        std::vector<Vertex>().swap(m_verts);
        std::vector<uint32_t>().swap(m_indices);

        m_uploaded = true;
        log::info("VR: batch ter-upload ({} quad, {} chunk, culling {})",
            m_quadCount, m_chunks.size(), (m_gpuCull && s_cullProg) ? "GPU" : "CPU");
        return true;
    }

    void updateChannels() {
        unsigned int frame = CCDirector::get()->getTotalFrames();
        if (frame == s_chanFrame) return;
        s_chanFrame = frame;

        std::memset(s_chanData, 255, sizeof(s_chanData));
        if (m_layer && m_layer->m_effectManager) {
            for (auto sprite : m_layer->m_effectManager->m_colorActionSpriteVector) {
                if (!sprite) continue;
                int id = sprite->m_colorID;
                if (id <= 0 || id >= kChannelCount) continue;
                float o = sprite->m_opacity;
                if (o <= 1.0f) o *= 255.f;
                o = std::min(std::max(o, 0.f), 255.f);
                s_chanData[id * 4 + 0] = sprite->m_color.r;
                s_chanData[id * 4 + 1] = sprite->m_color.g;
                s_chanData[id * 4 + 2] = sprite->m_color.b;
                s_chanData[id * 4 + 3] = (uint8_t)o;
            }
        }
        s_chanData[kBlackChannel * 4 + 0] = 0;
        s_chanData[kBlackChannel * 4 + 1] = 0;
        s_chanData[kBlackChannel * 4 + 2] = 0;
        s_chanData[kBlackChannel * 4 + 3] = 255;

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, s_chanTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kChannelCount, 1, GL_RGBA, GL_UNSIGNED_BYTE, s_chanData);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
    }

    static bool chunkVisible(Chunk const& c, float const* m, float margin) {
        float xs[2] = {c.minX, c.maxX};
        float ys[2] = {c.minY, c.maxY};
        float loX = 1e30f, hiX = -1e30f, loY = 1e30f, hiY = -1e30f;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                float x = m[0] * xs[i] + m[4] * ys[j] + m[12];
                float y = m[1] * xs[i] + m[5] * ys[j] + m[13];
                float w = m[3] * xs[i] + m[7] * ys[j] + m[15];
                if (w <= 0.f) return true;
                x /= w; y /= w;
                loX = std::min(loX, x); hiX = std::max(hiX, x);
                loY = std::min(loY, y); hiY = std::max(hiY, y);
            }
        }
        float lim = 1.f + margin;
        return hiX >= -lim && loX <= lim && hiY >= -lim && loY <= lim;
    }

    void cullCpu(float const* mvp, float margin, std::vector<std::pair<uint32_t, uint32_t>>& runs) {
        bool open = false;
        uint32_t first = 0, count = 0;
        for (auto& c : m_chunks) {
            if (chunkVisible(c, mvp, margin)) {
                if (!open) { open = true; first = c.first; count = 0; }
                count += c.count;
            } else if (open) {
                runs.emplace_back(first, count);
                open = false;
            }
        }
        if (open) runs.emplace_back(first, count);
    }

    void finishFrameStats(int drawnQuads) {
        int frame = (int)CCDirector::get()->getTotalFrames();
        if (frame != s_frame) {
            s_frame = frame;
            s_drawnLast = s_drawnAcc;
            s_drawnAcc = 0;
        }
        if (drawnQuads > 0) s_drawnAcc += drawnQuads;

        if (m_label && m_debug && frame % 20 == 0) {
            auto const& caps = vgl::caps();
            auto text = fmt::format(
                "Vinz Renderer\n{}\nGPU: {}\nbaked: {} quad\n{}",
                vgl::tierName(caps.tier), caps.renderer, m_quadCount,
                (m_gpuCull && s_cullProg) ? "cull: GPU (compute)"
                                          : fmt::format("cull: CPU, drawn ~{} quad", s_drawnLast)
            );
            m_label->setString(text.c_str());
        }
    }
};

} // namespace

bool isBaked(GameObject* obj) {
    return !g_baked.empty() && g_baked.find(obj) != g_baked.end();
}

void attach(PlayLayer* layer) {
    auto cfg = readConfig();
    if (!cfg.enabled || !layer || !layer->m_objects) return;

    if (!vgl::init()) {
        log::warn("GPU tidak mendukung OpenGL ES 3.0+ / GL 3.3+, renderer dimatikan");
        return;
    }
    auto const& caps = vgl::caps();

    bool gpuCull = false;
    if (cfg.mode == "OpenGL ES 3.1 (GPU culling)") {
        gpuCull = caps.tier == vgl::Tier::ES31;
        if (!gpuCull) log::warn("Mode ES 3.1 diminta tapi GPU tidak mendukung, pakai CPU culling");
    } else if (cfg.mode == "Auto") {
        gpuCull = caps.tier == vgl::Tier::ES31;
    }

    if (!buildPrograms(gpuCull)) {
        log::error("Shader renderer gagal dibuat, renderer dimatikan");
        return;
    }
    ensureChannelTexture();

    struct Group {
        std::vector<Quad> quads;
        std::vector<GameObject*> objs;
    };
    std::unordered_map<CCSpriteBatchNode*, Group> groups;

    size_t total = 0, baked = 0;
    for (auto obj : CCArrayExt<GameObject*>(layer->m_objects)) {
        total++;
        if (!isEligible(obj, cfg)) continue;
        std::vector<Quad> local;
        if (!bakeObject(obj, local) || local.empty()) continue;
        auto batch = static_cast<CCSpriteBatchNode*>(obj->getParent());
        auto& g = groups[batch];
        g.quads.insert(g.quads.end(), local.begin(), local.end());
        g.objs.push_back(obj);
        baked++;
    }

    CCLabelBMFont* label = nullptr;
    if (cfg.debug) {
        label = CCLabelBMFont::create("Vinz Renderer", "chatFont.fnt");
        label->setAnchorPoint({0.f, 1.f});
        label->setScale(0.5f);
        label->setPosition({4.f, CCDirector::get()->getWinSize().height - 4.f});
        layer->addChild(label, 9999);
    }

    bool first = true;
    for (auto& [batch, g] : groups) {
        auto parent = batch->getParent();
        if (!parent || g.quads.empty()) continue;
        auto objs = g.objs;
        auto node = BatchNode::create(batch, g.quads, std::move(objs), cfg, layer, gpuCull);
        if (!node) continue;
        if (first && label) { node->setLabel(label); first = false; }
        for (auto o : g.objs) {
            g_baked.insert(o);
            o->setVisible(false);
        }
        parent->addChild(node, batch->getZOrder());
    }

    log::info("Vinz Renderer: {} dari {} object di-bake", baked, total);
}

} // namespace vr
