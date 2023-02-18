#![feature(iterator_try_collect)]

#[cfg(all(target_os = "windows", target_env = "gnu"))]
#[link(name = "gcc_s", kind = "dylib")]
extern "C" {}

mod from_hdf5;

use nu_plugin::{serve_plugin, EvaluatedCall, LabeledError, MsgPackSerializer, Plugin};
use nu_protocol::{Category, Signature, Type, Value};

struct FromHdf5;

impl Plugin for FromHdf5 {
    fn signature(&self) -> Vec<Signature> {
        vec![Signature::build("from hdf5")
            .usage("Convert from HDF5 binary into table")
            .allow_variants_without_examples(true)
            .input_output_types(vec![(Type::Binary, Type::Any)])
            .category(Category::Experimental)
            .filter()]
    }

    fn run(
        &mut self,
        name: &str,
        call: &EvaluatedCall,
        input: &Value,
    ) -> Result<Value, LabeledError> {
        assert_eq!(name, "from hdf5");
        match input {
            Value::Binary { val, span } => {
                from_hdf5::from_hdf5_bytes(val, *span).map_err(|e| LabeledError {
                    label: "HDF5 error".into(),
                    msg: e.to_string(),
                    span: Some(call.head),
                })
            }
            v => Err(LabeledError {
                label: "Expected binary from pipeline".into(),
                msg: format!("requires binary input, got {}", v.get_type()),
                span: Some(call.head),
            }),
        }
    }
}

fn main() {
    serve_plugin(&mut FromHdf5, MsgPackSerializer);
}
