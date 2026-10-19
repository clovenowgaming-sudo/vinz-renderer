#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/GameObject.hpp>
#include "Renderer.hpp"

using namespace geode::prelude;

class $modify(VRPlayLayer, PlayLayer) {
    void setupHasCompleted() {
        PlayLayer::setupHasCompleted();
        vr::attach(this);
    }
};

// GD menyalakan visibility object tiap frame (updateVisibility).
// Object yang sudah di-bake harus tetap tersembunyi supaya tidak tergambar dua kali.
class $modify(VRGameObject, GameObject) {
    void setVisible(bool v) {
        if (v && vr::isBaked(this)) v = false;
        GameObject::setVisible(v);
    }
};
