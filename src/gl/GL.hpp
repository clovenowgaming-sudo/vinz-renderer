#pragma once
#include <Geode/Geode.hpp>
#include <string>

// Loader + deteksi kemampuan GL (OpenGL ES 3.0 / 3.1 / desktop 3.3+).
// Fungsi GLES2 biasa (glBindBuffer, glDrawElements, dst) dipakai langsung dari header cocos.
// Fungsi yang cuma ada di ES3/ES3.1 di-load manual di sini.

#ifndef GL_COMPUTE_SHADER
#define GL_COMPUTE_SHADER 0x91B9
#endif
#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_DRAW_INDIRECT_BUFFER
#define GL_DRAW_INDIRECT_BUFFER 0x8F3F
#endif
#ifndef GL_COMMAND_BARRIER_BIT
#define GL_COMMAND_BARRIER_BIT 0x00000040
#endif
#ifndef GL_DYNAMIC_COPY
#define GL_DYNAMIC_COPY 0x88EA
#endif

namespace vgl {
    enum class Tier { None, ES30, ES31, Desktop };

    struct Caps {
        Tier tier = Tier::None;
        bool gles = false;
        int major = 0;
        int minor = 0;
        bool compute = false; // compute + SSBO + indirect draw tersedia
        std::string version, vendor, renderer;
    };

    // Harus dipanggil saat GL context aktif (di thread render).
    bool init();
    Caps const& caps();
    char const* tierName(Tier t);

    // Wrapper fungsi ES3 / ES3.1
    void genVertexArrays(int n, unsigned int* ids);
    void deleteVertexArrays(int n, unsigned int const* ids);
    void bindVertexArray(unsigned int id);
    void vertexAttribIPointer(unsigned int idx, int size, unsigned int type, int stride, void const* ptr);
    void bindBufferBase(unsigned int target, unsigned int index, unsigned int buffer);
    void dispatchCompute(unsigned int x, unsigned int y, unsigned int z);
    void memoryBarrier(unsigned int bits);
    void drawElementsIndirect(unsigned int mode, unsigned int type, void const* indirect);

    // Shader helper. Mengembalikan 0 kalau gagal (log error ditulis ke log Geode).
    std::string glslHeader(bool compute);
    unsigned int compileShader(unsigned int type, std::string const& source, char const* name);
    unsigned int linkProgram(unsigned int a, unsigned int b = 0);
}
