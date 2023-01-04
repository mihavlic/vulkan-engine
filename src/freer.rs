use crate::object;

pub enum DeadResource {
    Image(object::Image),
    Buffer(object::Buffer),
}

pub struct Freer {}
