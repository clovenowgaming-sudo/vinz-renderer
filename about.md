# Vinz Renderer (experimental)

Renderer GPU eksperimental buat GD 2.2081, jalan di Android (OpenGL ES 3.0 / 3.1) dan Windows.

**Masih alpha.** Kalau ada level yang kelihatan rusak, matiin "Renderer aktif" di settings lalu masuk ulang level.

- Object statis (dekorasi / solid tanpa group) di-bake sekali ke buffer GPU
- Warna dibaca di GPU dari color channel, jadi color/pulse trigger tetap jalan
- ES 3.0: culling di CPU per chunk
- ES 3.1: culling pakai compute shader + glDrawElementsIndirect
- Object yang bisa digerakkan trigger tetap digambar oleh GD seperti biasa
