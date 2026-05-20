# Tugas Internet untuk Segala Komputasi - Simulasi Particle Swarm Optimization (PSO)

Repository ini berisi kode program dan hasil simulasi Particle Swarm Optimization (PSO) untuk memenuhi tugas lab report pada mata kuliah Internet untuk Segala Komputasi. Simulasi dilakukan berdasarkan fungsi objektif :

\[
f(x) = x^2
\]

Pada fungsi tersebut, titik optimum atau minimum global berada pada \(x = 0\) dan \(f(x) = 0\). Simulasi ini berfokus pada analisis pengaruh nilai inertia weight \(w\) terhadap jumlah iterasi yang dibutuhkan PSO untuk mencapai kondisi target.

- Nama: Marcelia Chintya
- NIM: 1101223073
- Mata Kuliah: Internet untuk Segala Komputasi

## Deskripsi Tugas

Berdasarkan script Python PSO yang dibahas di kelas, tugas ini bertujuan untuk melakukan simulasi dan menganalisis salah satu parameter PSO. Pada tugas ini, simulasi yang dipilih adalah:

> Bagaimana hubungan konstanta \(w\) terhadap jumlah iterasi yang dibutuhkan untuk mencapai status 90% partikel berada pada posisi kurang dari 10% dari posisi optimum.

Karena titik optimum fungsi \(f(x)=x^2\) berada pada \(x=0\), maka batas 10% dari posisi optimum tidak dapat dihitung secara langsung. Oleh karena itu, pada simulasi ini digunakan definisi operasional berupa target zone.
Target dianggap tercapai apabila minimal 90% partikel berada di dalam zona tersebut.

## Parameter Simulasi

Parameter dasar yang digunakan dalam simulasi adalah sebagai berikut:

| Parameter | Nilai |
|---|---|
| Fungsi objektif | \(f(x)=x^2\) |
| Jumlah partikel | 20 |
| Maksimum iterasi | 100 |
| Cognitive coefficient \(c1\) | 1.5 |
| Social coefficient \(c2\) | 1.5 |
| Target rasio partikel | 90% |
| Jumlah percobaan per nilai \(w\) | 30 |

Nilai inertia weight \(w\) yang diuji:

```text
0.1, 0.3, 0.5, 0.7, 0.9, 1.1
