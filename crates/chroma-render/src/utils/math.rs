#[allow(dead_code)]
pub fn div_up<
    T: std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + From<i32>
        + Copy,
>(
    a: T,
    b: T,
) -> T {
    (a + b - T::from(1)) / b
}
