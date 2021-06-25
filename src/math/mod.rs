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

#[inline]
pub fn copy_u8_to_i32(src: &[u8], dest: &mut [i32]) {
    let dest = &mut dest[0..src.len()]; // eliminates a branch from the loop
    for (value, dest) in src.iter().copied().zip(dest.iter_mut()) {
        *dest = i32::from(value);
    }
}

pub fn square(src: &[i32], dest: &mut [u32]) {
    for (value, dest) in src.iter().copied().zip(dest.iter_mut()) {
        *dest = i32::pow(value, 2) as u32;
    }
}

pub unsafe fn abs(src: *const i32, dest: *mut i32, length: usize) {
    for i in 0..length as isize {
        let value = *src.offset(i);
        *dest.offset(i) = if value >= 0 { value } else { -value };
    }
}

pub unsafe fn vector_add(left: *const i32, right: *const i32, dest: *mut i32, length: usize) {
    for i in 0..length as isize {
        *dest.offset(i) = *left.offset(i) + *right.offset(i);
    }
}

pub unsafe fn vector_sub(left: *const i32, right: *const i32, dest: *mut i32, length: usize) {
    for i in 0..length as isize {
        *dest.offset(i) = *left.offset(i) - *right.offset(i);
    }
}

pub fn vector_inner_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(l, r)| l * r)
        .sum()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_square() {
        let mut vec = vec![1, 2, 3];
        square(&[1, 2, 3], &mut vec);
        assert_eq!(vec![1, 4, 9], vec);
    }

    #[test]
    fn test_abs() {
        let mut vec = vec![-1, 2, -3];
        unsafe { abs(vec.as_ptr(), vec.as_mut_ptr(), vec.len()) };
        assert_eq!(vec![1, 2, 3], vec);
    }

    #[test]
    fn test_vector_add() {
        let mut vec = vec![1, 2, 3];
        unsafe { vector_add(vec.as_ptr(), vec.as_ptr(), vec.as_mut_ptr(), vec.len()) };
        assert_eq!(vec![2, 4, 6], vec);
    }

    #[test]
    fn test_vector_sub() {
        let mut vec = vec![1, 2, 3];
        unsafe { vector_sub(vec.as_ptr(), vec.as_ptr(), vec.as_mut_ptr(), vec.len()) };
        assert_eq!(vec![0, 0, 0], vec);
    }

    #[test]
    fn test_vector_inner_product() {
        let vec = vec![1.0, 2.0, 3.0];
        let result = vector_inner_product(&vec, &vec);
        assert_eq!(14.0, result);
    }
}
