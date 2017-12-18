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