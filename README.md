# Vinz Renderer (experimental)

Mod Geode buat Geometry Dash 2.2081 (Geode SDK 5.11.2): Android 32/64 + Windows.
Kode original, ditulis dari nol. Terinspirasi dari ide umum "render object lewat GPU", bukan salinan mod lain.

## Cara kerja
1. `PlayLayer::setupHasCompleted` -> object statis di-bake (geometri 4 vertex/sprite) ke VBO.
2. Vertex hanya simpan nomor color channel; warna dibaca di vertex shader dari tekstur 1024x1
   yang di-update 1x per frame dari `GJEffectManager`.
3. Quad dipotong per chunk (sumbu X). Chunk yang terlihat = satu rentang index yang menyambung.
   - ES 3.0 / desktop: cek di CPU, `glDrawElements`
   - ES 3.1: compute shader `vr.cull.comp` -> `glDrawElementsIndirect`
4. Sprite asli disembunyikan lewat hook `GameObject::setVisible`.

## Batasan (alpha)
- Cuma object dekorasi/solid tanpa group, tanpa HSV, tanpa animasi yang di-bake.
  Object yang bisa digerakkan, di-toggle, atau di-alpha trigger tetap digambar GD biasa.
- Urutan z antara object yang di-bake dan yang tidak, di dalam satu batch node yang sama, bisa sedikit beda.
- Glow, particle, dan shader level tidak disentuh.
- Belum ada pembuktian FPS naik: lihat log dan overlay debug dulu.

## Build
Push ke GitHub, buka tab Actions, download artifact "Build Output".
