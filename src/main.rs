#![feature(c_size_t)]
#![feature(iterator_try_collect)]

#[cfg(all(target_os = "windows", target_env = "gnu", not(debug_assertions)))]
#[link(name = "gcc_s", kind = "dylib")]
extern "C" {}

mod from_hdf5;
mod hdf5_ext;

use nu_plugin::{serve_plugin, EvaluatedCall, LabeledError, MsgPackSerializer, Plugin};
use nu_protocol::{PluginSignature, Value};

struct FromHdf5;

impl Plugin for FromHdf5 {
    fn signature(&self) -> Vec<PluginSignature> {
        vec![from_hdf5::signature()]
    }

    fn run(
        &mut self,
        name: &str,
        call: &EvaluatedCall,
        input: &Value,
    ) -> Result<Value, LabeledError> {
        match name {
            "from hdf5" => from_hdf5::run(call, input),
            _ => unreachable!(),
        }
    }
}

fn main() {
    serve_plugin(&mut FromHdf5, MsgPackSerializer);
}
