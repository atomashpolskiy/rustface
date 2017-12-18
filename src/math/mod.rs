pub unsafe fn copy_u8_to_i32(src: *const u8, dest: *mut i32, length: usize) {
    for i in 0..length as isize {
        *dest.offset(i) = *src.offset(i) as i32;
    }
}

pub unsafe fn square(src: *const i32, dest: *mut i32, length: usize) {
    for i in 0..length as isize {
        let value = *src.offset(i);
        *dest.offset(i) = value * value;
    }
}

pub unsafe fn vector_add(left: *const i32, right: *const i32, dest: *mut i32, length: usize) {
    apply_for_range(
        &mut |x, y, z| *z = *x + *y,
        left, right, dest, length
    );
}

pub unsafe fn vector_sub(left: *const i32, right: *const i32, dest: *mut i32, length: usize) {
    apply_for_range(
        &mut |x, y, z| *z = *x - *y,
        left, right, dest, length
    );
}

unsafe fn apply_for_range<F>(op: &mut F, left: *const i32, right: *const i32, dest: *mut i32, length: usize)
    where F: FnMut(*const i32, *const i32, *mut i32) {
    for i in 0..length as isize {
        apply(op, left.offset(i), right.offset(i), dest.offset(i));
    }
}

unsafe fn apply<F>(op: &mut F, left: *const i32, right: *const i32, dest: *mut i32)
    where F: FnMut(*const i32, *const i32, *mut i32) {
    op(left, right, dest);
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_vector_add() {
        let mut vec = vec![1, 2, 3];
        unsafe {
            vector_add(vec.as_ptr(), vec.as_ptr(), vec.as_mut_ptr(), vec.len());
        }
        assert_eq!(vec![2, 4, 6], vec)
    }

    #[test]
    fn test_vector_sub() {
        let mut vec = vec![1, 2, 3];
        unsafe {
            vector_sub(vec.as_ptr(), vec.as_ptr(), vec.as_mut_ptr(), vec.len());
        }
        assert_eq!(vec![0, 0, 0], vec)
    }
}