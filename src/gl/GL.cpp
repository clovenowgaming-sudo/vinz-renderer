#include "GL.hpp"
#include <cstdio>
#include <cstring>

#ifdef GEODE_IS_ANDROID
#include <dlfcn.h>
#endif

using namespace geode::prelude;

namespace vgl {
    namespace {
        Caps s_caps;
        bool s_init = false;

        typedef void (*PFN_GenVAO)(int, unsigned int*);
        typedef void (*PFN_DelVAO)(int, unsigned int const*);
        typedef void (*PFN_BindVAO)(unsigned int);
        typedef void (*PFN_AttribI)(unsigned int, int, unsigned int, int, void const*);
        typedef void (*PFN_BindBase)(unsigned int, unsigned int, unsigned int);
        typedef void (*PFN_Dispatch)(unsigned int, unsigned int, unsigned int);
        typedef void (*PFN_Barrier)(unsigned int);
        typedef void (*PFN_DrawInd)(unsigned int, unsigned int, void const*);

        PFN_GenVAO fnGenVAO = nullptr;
        PFN_DelVAO fnDelVAO = nullptr;
        PFN_BindVAO fnBindVAO = nullptr;
        PFN_AttribI fnAttribI = nullptr;
        PFN_BindBase fnBindBase = nullptr;
        PFN_Dispatch fnDispatch = nullptr;
        PFN_Barrier fnBarrier = nullptr;
        PFN_DrawInd fnDrawInd = nullptr;

        void* getProc(char const* name) {
#if defined(GEODE_IS_WINDOWS)
            void* p = (void*)wglGetProcAddress(name);
            if (!p || p == (void*)1 || p == (void*)2 || p == (void*)3 || p == (void*)-1) {
                static HMODULE lib = GetModuleHandleA("opengl32.dll");
                p = lib ? (void*)GetProcAddress(lib, name) : nullptr;
            }
            return p;
#elif defined(GEODE_IS_ANDROID)
            typedef void* (*GetProcFn)(char const*);
            static void* egl = dlopen("libEGL.so", RTLD_NOW);
            static void* gles3 = dlopen("libGLESv3.so", RTLD_NOW);
            static GetProcFn getp = egl ? (GetProcFn)dlsym(egl, "eglGetProcAddress") : nullptr;
            void* p = getp ? getp(name) : nullptr;
            if (!p && gles3) p = dlsym(gles3, name);
            return p;
#else
            return nullptr;
#endif
        }
    }

    bool init() {
        if (s_init) return s_caps.tier != Tier::None;
        s_init = true;

        auto ver = reinterpret_cast<char const*>(glGetString(GL_VERSION));
        auto ven = reinterpret_cast<char const*>(glGetString(GL_VENDOR));
        auto ren = reinterpret_cast<char const*>(glGetString(GL_RENDERER));
        if (!ver) {
            log::error("glGetString(GL_VERSION) gagal, GL context belum aktif?");
            return false;
        }
        s_caps.version = ver;
        s_caps.vendor = ven ? ven : "?";
        s_caps.renderer = ren ? ren : "?";

        int major = 0, minor = 0;
        if (std::strncmp(ver, "OpenGL ES", 9) == 0) {
            s_caps.gles = true;
            std::sscanf(ver, "OpenGL ES %d.%d", &major, &minor);
        } else {
            std::sscanf(ver, "%d.%d", &major, &minor);
        }
        s_caps.major = major;
        s_caps.minor = minor;

        fnGenVAO = (PFN_GenVAO)getProc("glGenVertexArrays");
        fnDelVAO = (PFN_DelVAO)getProc("glDeleteVertexArrays");
        fnBindVAO = (PFN_BindVAO)getProc("glBindVertexArray");
        fnAttribI = (PFN_AttribI)getProc("glVertexAttribIPointer");
        bool core3 = fnGenVAO && fnDelVAO && fnBindVAO && fnAttribI;

        if (s_caps.gles) {
            if (major >= 3 && core3) {
                s_caps.tier = Tier::ES30;
                if (major > 3 || minor >= 1) {
                    fnBindBase = (PFN_BindBase)getProc("glBindBufferBase");
                    fnDispatch = (PFN_Dispatch)getProc("glDispatchCompute");
                    fnBarrier = (PFN_Barrier)getProc("glMemoryBarrier");
                    fnDrawInd = (PFN_DrawInd)getProc("glDrawElementsIndirect");
                    s_caps.compute = fnBindBase && fnDispatch && fnBarrier && fnDrawInd;
                    if (s_caps.compute) s_caps.tier = Tier::ES31;
                }
            }
        } else {
            if ((major > 3 || (major == 3 && minor >= 3)) && core3)
                s_caps.tier = Tier::Desktop;
        }

        log::info("GL: {} | {} | {}", s_caps.version, s_caps.vendor, s_caps.renderer);
        log::info("Vinz Renderer tier: {} (compute: {})", tierName(s_caps.tier), s_caps.compute);
        return s_caps.tier != Tier::None;
    }

    Caps const& caps() { return s_caps; }

    char const* tierName(Tier t) {
        switch (t) {
            case Tier::ES30: return "OpenGL ES 3.0";
            case Tier::ES31: return "OpenGL ES 3.1";
            case Tier::Desktop: return "OpenGL Desktop 3.3+";
            default: return "unsupported";
        }
    }

    void genVertexArrays(int n, unsigned int* ids) { fnGenVAO(n, ids); }
    void deleteVertexArrays(int n, unsigned int const* ids) { if (fnDelVAO) fnDelVAO(n, ids); }
    void bindVertexArray(unsigned int id) { fnBindVAO(id); }
    void vertexAttribIPointer(unsigned int idx, int size, unsigned int type, int stride, void const* ptr) {
        fnAttribI(idx, size, type, stride, ptr);
    }
    void bindBufferBase(unsigned int target, unsigned int index, unsigned int buffer) { fnBindBase(target, index, buffer); }
    void dispatchCompute(unsigned int x, unsigned int y, unsigned int z) { fnDispatch(x, y, z); }
    void memoryBarrier(unsigned int bits) { fnBarrier(bits); }
    void drawElementsIndirect(unsigned int mode, unsigned int type, void const* indirect) { fnDrawInd(mode, type, indirect); }

    std::string glslHeader(bool compute) {
        switch (s_caps.tier) {
            case Tier::ES31:
                return compute
                    ? "#version 310 es\nprecision highp float;\nprecision highp int;\n"
                    : "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n";
            case Tier::ES30:
                return "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n";
            case Tier::Desktop:
                return "#version 330 core\n";
            default:
                return "";
        }
    }

    unsigned int compileShader(unsigned int type, std::string const& source, char const* name) {
        GLuint sh = glCreateShader(type);
        char const* src = source.c_str();
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char buf[2048] = {0};
            glGetShaderInfoLog(sh, sizeof(buf) - 1, nullptr, buf);
            log::error("Shader '{}' gagal compile: {}", name, buf);
            glDeleteShader(sh);
            return 0;
        }
        return sh;
    }

    unsigned int linkProgram(unsigned int a, unsigned int b) {
        if (!a) return 0;
        GLuint prog = glCreateProgram();
        glAttachShader(prog, a);
        if (b) glAttachShader(prog, b);
        glLinkProgram(prog);
        GLint ok = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        glDeleteShader(a);
        if (b) glDeleteShader(b);
        if (!ok) {
            char buf[2048] = {0};
            glGetProgramInfoLog(prog, sizeof(buf) - 1, nullptr, buf);
            log::error("Program gagal link: {}", buf);
            glDeleteProgram(prog);
            return 0;
        }
        return prog;
    }
}
