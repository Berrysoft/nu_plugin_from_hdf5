#![feature(c_size_t)]
#![feature(iterator_try_collect)]

mod from_hdf5;
mod hdf5_ext;

use nu_plugin::{
    serve_plugin, EngineInterface, EvaluatedCall, MsgPackSerializer, Plugin, PluginCommand,
};
use nu_protocol::{LabeledError, PipelineData, Signature};

struct FromHdf5;

impl Plugin for FromHdf5 {
    fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    fn commands(&self) -> Vec<Box<dyn PluginCommand<Plugin = Self>>> {
        vec![Box::new(FromHdf5)]
    }
}

impl PluginCommand for FromHdf5 {
    type Plugin = FromHdf5;

    fn name(&self) -> &str {
        "from hdf5"
    }

    fn usage(&self) -> &str {
        "Read HDF5 file and output a table"
    }

    fn signature(&self) -> Signature {
        from_hdf5::signature()
    }

    fn run(
        &self,
        _plugin: &FromHdf5,
        _engine: &EngineInterface,
        _call: &EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        from_hdf5::run(input)
    }
}

fn main() {
    serve_plugin(&FromHdf5, MsgPackSerializer);
}
