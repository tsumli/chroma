fn main() {
    // compile shaders
    chroma_shader::command::compile_all().expect("failed to compile shaders");
}
