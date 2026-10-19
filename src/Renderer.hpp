#pragma once
#include <Geode/Geode.hpp>

namespace vr {
    // Dipanggil setelah PlayLayer selesai setup: bake object statis dan pasang node renderer.
    void attach(PlayLayer* layer);

    // True kalau object sudah di-bake (jangan biarkan GD menggambarnya lagi).
    bool isBaked(GameObject* obj);
}
