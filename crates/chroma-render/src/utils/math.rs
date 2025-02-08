use num_traits::Num;

pub fn div_up<T: Num + Copy>(a: T, b: T) -> T {
    (a + b - T::one()) / b
}

pub fn align_up<T: Num + Copy>(a: T, b: T) -> T {
    div_up(a, b) * b
}
