// This file is part of the open-source port of SeetaFace engine, which originally includes three modules:
//      SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
//
// This file is part of the SeetaFace Detection module, containing codes implementing the face detection method described in the following paper:
//
//      Funnel-structured cascade for multi-view face detection with alignment awareness,
//      Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
//      In Neurocomputing (under review)
//
// Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
// Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
//
// As an open-source face recognition engine: you can redistribute SeetaFace source codes
// and/or modify it under the terms of the BSD 2-Clause License.
//
// You should have received a copy of the BSD 2-Clause License along with the software.
// If not, see < https://opensource.org/licenses/BSD-2-Clause>.

use stdsimd::simd::{i32x4, i32x8, f32x4};
use stdsimd::vendor::{__m128i, __m256i,
                      _mm_setzero_ps, _mm_loadu_ps, _mm_storeu_ps, _mm_add_ps, _mm_mul_ps,
                      _mm_add_epi32, _mm_sub_epi32, _mm_abs_epi32, _mm256_mullo_epi32,
                      _mm_loadu_si128, _mm256_loadu_si256, _mm_storeu_si128, _mm256_storeu_si256};

pub fn copy_u8_to_i32(src: *const u8, dest: *mut i32, length: usize) {
    unsafe {
        for i in 0..length as isize {
            *dest.offset(i) = i32::from(*src.offset(i));
        }
    }
}

pub fn square(src: *const i32, dest: *mut u32, length: usize) {
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "avx2"))]
        {
            unsafe {
                square_avx2(src, dest, length);
            }
        }
    #[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "avx2")))]
        {
            square_portable(src, dest, length);
        }
}

#[allow(unused)]
fn square_portable(src: *const i32, dest: *mut u32, length: usize) {
    unsafe {
        for i in 0..length as isize {
            let value = *src.offset(i);
            *dest.offset(i) = i32::pow(value, 2) as u32;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature = "+avx2"]
#[allow(unused)]
unsafe fn square_avx2(src: *const i32, dest: *mut u32, length: usize) {
    let mut i: isize = 0;

    let mut x1: __m256i;
    let mut x2 = src as *const __m256i;
    let mut z2 = dest as *mut __m256i;

    // _mm_mullo_epi32 is not supported in Rust yet, see https://github.com/rust-lang-nursery/stdsimd/issues/40
    // might use _mm_mullo_epi16 (SSE2) for better portability instead, because inputs are unlikely to exceed int16
    while i < (length as isize - 8) {
        x1 = _mm256_loadu_si256(x2);
        _mm256_storeu_si256(z2, __m256i::from(_mm256_mullo_epi32(i32x8::from(x1), i32x8::from(x1))));

        x2 = x2.offset(1);
        z2 = z2.offset(1);
        i += 8;
    }

    for _ in i..(length as isize) {
        let value = *src.offset(i);
        *dest.offset(i) = i32::pow(value, 2) as u32;
    }
}

pub fn abs(src: *const i32, dest: *mut i32, length: usize) {
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "ssse3"))]
        {
            unsafe {
                abs_ssse3(src, dest, length);
            }
        }
    #[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "ssse3")))]
        {
            abs_portable(src, dest, length);
        }
}

#[allow(unused)]
fn abs_portable(src: *const i32, dest: *mut i32, length: usize) {
    unsafe {
        for i in 0..length as isize {
            let value = *src.offset(i);
            *dest.offset(i) = if value >= 0 { value } else { -value };
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature = "+ssse3"]
#[allow(unused)]
unsafe fn abs_ssse3(src: *const i32, dest: *mut i32, length: usize) {
    let mut i: isize = 0;

    let mut val: __m128i;
    let mut val_abs: __m128i;

    let mut x = src as *const __m128i;
    let mut z = dest as *mut __m128i;

    while i < (length as isize - 4) {
        val = _mm_loadu_si128(x);
        val_abs = __m128i::from(_mm_abs_epi32(i32x4::from(val)));
        _mm_storeu_si128(z, val_abs);

        x = x.offset(1);
        z = z.offset(1);
        i += 4;
    }

    for k in i..(length as isize) {
        let value = *src.offset(k as isize);
        *dest.offset(k as isize) = if value >= 0 { value } else { -value };
    }
}

pub fn vector_add(left: *const i32, right: *const i32, dest: *mut i32, length: usize) {
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse2"))]
        {
            unsafe {
                vector_add_sse2(left, right, dest, length);
            }
        }
    #[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse2")))]
        {
            vector_add_portable(left, right, dest, length);
        }
}

#[allow(unused)]
fn vector_add_portable(left: *const i32, right: *const i32, dest: *mut i32, length: usize) {
    unsafe {
        for i in 0..length as isize {
            *dest.offset(i) = *left.offset(i) + *right.offset(i);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature = "+sse2"]
#[allow(unused)]
unsafe fn vector_add_sse2(left: *const i32, right: *const i32, dest: *mut i32, length: usize) {
    let mut i: isize = 0;

    let mut x1: __m128i;
    let mut y1: __m128i;

    let mut x2 = left as *const __m128i;
    let mut y2 = right as *const __m128i;
    let mut z2 = dest as *mut __m128i;

    while i < (length as isize - 4) {
        x1 = _mm_loadu_si128(x2);
        y1 = _mm_loadu_si128(y2);
        _mm_storeu_si128(z2, __m128i::from(_mm_add_epi32(i32x4::from(x1), i32x4::from(y1))));

        x2 = x2.offset(1);
        y2 = y2.offset(1);
        z2 = z2.offset(1);
        i += 4;
    }

    for k in i..(length as isize) {
        *dest.offset(k) = *left.offset(k) + *right.offset(k);
    }
}

pub fn vector_sub(left: *const i32, right: *const i32, dest: *mut i32, length: usize) {
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse2"))]
        {
            unsafe {
                vector_sub_sse2(left, right, dest, length);
            }
        }
    #[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse2")))]
        {
            vector_sub_portable(left, right, dest, length);
        }
}

#[allow(unused)]
fn vector_sub_portable(left: *const i32, right: *const i32, dest: *mut i32, length: usize) {
    unsafe {
        for i in 0..length as isize {
            *dest.offset(i) = *left.offset(i) - *right.offset(i);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature = "+sse2"]
#[allow(unused)]
unsafe fn vector_sub_sse2(left: *const i32, right: *const i32, dest: *mut i32, length: usize) {
    let mut i: isize = 0;

    let mut x1: __m128i;
    let mut y1: __m128i;

    let mut x2 = left as *const __m128i;
    let mut y2 = right as *const __m128i;
    let mut z2 = dest as *mut __m128i;

    while i < (length as isize - 4) {
        x1 = _mm_loadu_si128(x2);
        y1 = _mm_loadu_si128(y2);
        _mm_storeu_si128(z2, __m128i::from(_mm_sub_epi32(i32x4::from(x1), i32x4::from(y1))));

        x2 = x2.offset(1);
        y2 = y2.offset(1);
        z2 = z2.offset(1);
        i += 4;
    }

    for k in i..(length as isize) {
        *dest.offset(k) = *left.offset(k) - *right.offset(k);
    }
}

pub fn vector_inner_product(left: *const f32, right: *const f32, length: usize) -> f32 {
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse"))]
        {
            unsafe {
                vector_inner_product_sse(left, right, length)
            }
        }
    #[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse")))]
        {
            vector_inner_product_portable(left, right, length)
        }
}

#[allow(unused)]
fn vector_inner_product_portable(left: *const f32, right: *const f32, length: usize) -> f32 {
    let mut product = 0.0;
    unsafe {
        for i in 0..length as isize {
            product += (*left.offset(i)) * (*right.offset(i));
        }
    }
    product
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature = "+sse"]
#[allow(unused)]
unsafe fn vector_inner_product_sse(left: *const f32, right: *const f32, length: usize) -> f32 {
    let mut product;
    let mut i: isize = 0;

    let mut x1: f32x4;
    let mut y1: f32x4;
    let mut z1 = _mm_setzero_ps();
    let mut buf = vec![0.0; 4];

    while i < (length as isize - 4) {
        x1 = _mm_loadu_ps(left.offset(i));
        y1 = _mm_loadu_ps(right.offset(i));
        z1 = _mm_add_ps(z1, _mm_mul_ps(x1, y1));
        i += 4;
    }
    _mm_storeu_ps(buf.as_mut_ptr(), z1);
    product = buf[0] + buf[1] + buf[2] + buf[3];

    for k in i..(length as isize) {
        product += (*left.offset(k)) * (*right.offset(k));
    }
    product
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_square_portable() {
        let mut vec = vec![1, 2, 3];
        square_portable(vec.as_ptr(), vec.as_mut_ptr() as *mut u32, vec.len());
        assert_eq!(vec![1, 4, 9], vec);
    }

    #[test]
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "avx2"))]
    fn test_square_avx2() {
        let mut vec = vec![1, 2, 3, 4, -1, -2, -3, -4, 5, -6];
        unsafe {
            square_avx2(vec.as_ptr(), vec.as_mut_ptr() as *mut u32, vec.len());
        }
        assert_eq!(vec![1, 4, 9, 16, 1, 4, 9, 16, 25, 36], vec);
    }

    #[test]
    fn test_abs_portable() {
        let mut vec = vec![-1, 2, -3];
        abs_portable(vec.as_ptr(), vec.as_mut_ptr(), vec.len());
        assert_eq!(vec![1, 2, 3], vec);
    }

    #[test]
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "ssse3"))]
    fn test_abs_ssse3() {
        let mut vec = vec![-1, 2, -3, 4, -5, 6];
        unsafe {
            abs_ssse3(vec.as_ptr(), vec.as_mut_ptr(), vec.len());
        }
        assert_eq!(vec![1, 2, 3, 4, 5, 6], vec);
    }

    #[test]
    fn test_vector_add_portable() {
        let mut vec = vec![1, 2, 3];
        vector_add_portable(vec.as_ptr(), vec.as_ptr(), vec.as_mut_ptr(), vec.len());
        assert_eq!(vec![2, 4, 6], vec);
    }

    #[test]
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse2"))]
    fn test_vector_add_sse2() {
        let mut vec = vec![1, 2, 3, 4, -1, -2, -3, -4, -5, 6];
        unsafe {
            vector_add_sse2(vec.as_ptr(), vec.as_ptr(), vec.as_mut_ptr(), vec.len());
        }
        assert_eq!(vec![2, 4, 6, 8, -2, -4, -6, -8, -10, 12], vec);
    }

    #[test]
    fn test_vector_sub_portable() {
        let mut vec = vec![1, 2, 3];
        vector_sub_portable(vec.as_ptr(), vec.as_ptr(), vec.as_mut_ptr(), vec.len());
        assert_eq!(vec![0, 0, 0], vec);
    }

    #[test]
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse2"))]
    fn test_vector_sub_sse2() {
        let mut vec = vec![1, 2, 3, 4, -1, -2, -3, -4, -5, 6];
        unsafe {
            vector_sub_sse2(vec.as_ptr(), vec.as_ptr(), vec.as_mut_ptr(), vec.len());
        }
        assert_eq!(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], vec);
    }

    #[test]
    fn test_vector_inner_product_portable() {
        let vec = vec![1.0, 2.0, 3.0];
        let result = vector_inner_product_portable(vec.as_ptr(), vec.as_ptr(), vec.len());
        assert_eq!(14.0, result);
    }

    #[test]
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse"))]
    fn test_vector_inner_product_sse() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = unsafe {
            vector_inner_product_sse(vec.as_ptr(), vec.as_ptr(), vec.len())
        };
        assert_eq!(91.0, result);
    }
}