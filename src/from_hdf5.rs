use crate::hdf5_ext::{FileImage, ReadRawBytes};
use hdf5::{
    types::{FloatSize, IntSize, TypeDescriptor, VarLenArray, VarLenAscii, VarLenUnicode},
    Dataset, Group, Result,
};
use nu_plugin::{EvaluatedCall, LabeledError};
use nu_protocol::{Category, Signature, Span, Type, Value};

macro_rules! native {
    ($native_ty: ty, $slice: expr) => {
        $slice.as_ptr().cast::<$native_ty>()
    };
}

macro_rules! native_value {
    ($native_ty: ty, $variant: ident, $slice: expr, $span: expr) => {{
        assert_eq!($slice.len(), std::mem::size_of::<$native_ty>());
        Value::$variant {
            val: unsafe { std::ptr::read_unaligned(native!($native_ty, $slice)) } as _,
            span: $span,
        }
    }};
}

fn to_value(slice: &[u8], dtype: &TypeDescriptor, span: Span) -> Result<Value> {
    let val = match dtype {
        TypeDescriptor::Integer(IntSize::U1) => native_value!(i8, Int, slice, span),
        TypeDescriptor::Integer(IntSize::U2) => native_value!(i16, Int, slice, span),
        TypeDescriptor::Integer(IntSize::U4) => native_value!(i32, Int, slice, span),
        TypeDescriptor::Integer(IntSize::U8) => native_value!(i64, Int, slice, span),
        TypeDescriptor::Unsigned(IntSize::U1) => native_value!(u8, Int, slice, span),
        TypeDescriptor::Unsigned(IntSize::U2) => native_value!(u16, Int, slice, span),
        TypeDescriptor::Unsigned(IntSize::U4) => native_value!(u32, Int, slice, span),
        TypeDescriptor::Unsigned(IntSize::U8) => native_value!(u64, Int, slice, span),
        TypeDescriptor::Float(FloatSize::U4) => native_value!(f32, Float, slice, span),
        TypeDescriptor::Float(FloatSize::U8) => native_value!(f64, Float, slice, span),
        TypeDescriptor::Boolean => native_value!(bool, Bool, slice, span),
        TypeDescriptor::Enum(ty) => {
            let int_size = ty.size;
            if ty.signed {
                to_value(slice, &TypeDescriptor::Integer(int_size), span)?
            } else {
                to_value(slice, &TypeDescriptor::Unsigned(int_size), span)?
            }
        }
        TypeDescriptor::Compound(comp) => {
            assert_eq!(slice.len(), comp.size);
            let mut cols = vec![];
            let mut vals = vec![];
            for field in comp.fields.iter() {
                cols.push(field.name.clone());
                vals.push(to_value(
                    &slice[field.offset..field.offset + field.ty.size()],
                    &field.ty,
                    span,
                )?)
            }
            Value::Record { cols, vals, span }
        }
        TypeDescriptor::FixedArray(ty, len) => {
            assert_eq!(slice.len(), ty.size() * len);
            Value::List {
                vals: slice
                    .chunks(ty.size())
                    .map(|slice| to_value(slice, ty, span))
                    .try_collect()?,
                span,
            }
        }
        TypeDescriptor::FixedAscii(len) | TypeDescriptor::FixedUnicode(len) => {
            assert_eq!(slice.len(), *len);
            Value::String {
                val: String::from_utf8_lossy(slice).into_owned(),
                span,
            }
        }
        TypeDescriptor::VarLenArray(ty) => {
            let hvl = unsafe { native!(VarLenArray<u8>, slice).as_ref() }.unwrap();
            Value::List {
                vals: hvl
                    .chunks(ty.size())
                    .map(|slice| to_value(slice, ty, span))
                    .try_collect()?,
                span,
            }
        }
        TypeDescriptor::VarLenAscii => {
            let str = unsafe { native!(VarLenAscii, slice).as_ref() }.unwrap();
            Value::String {
                val: str.as_str().to_string(),
                span,
            }
        }
        TypeDescriptor::VarLenUnicode => {
            let str = unsafe { native!(VarLenUnicode, slice).as_ref() }.unwrap();
            Value::String {
                val: str.as_str().to_string(),
                span,
            }
        }
    };
    Ok(val)
}

fn to_list(dataset: &Dataset, span: Span) -> Result<Value> {
    let dtype = dataset.dtype()?.to_descriptor()?;
    let data = dataset.read_raw_bytes(&dtype)?;
    let vals: Vec<Value> = data
        .chunks(dtype.size())
        .map(|slice| to_value(slice, &dtype, span))
        .try_collect()?;
    assert_eq!(vals.len(), dataset.size());
    Ok(Value::List { vals, span })
}

fn to_record(group: &Group, span: Span) -> Result<Value> {
    let mut cols = vec![];
    let mut vals = vec![];
    for ds in group.datasets()? {
        cols.push(strip_name(ds.name()));
        vals.push(to_list(&ds, span)?);
    }
    for g in group.groups()? {
        cols.push(strip_name(g.name()));
        vals.push(to_record(&g, span)?);
    }
    Ok(Value::Record { cols, vals, span })
}

fn strip_name(name: String) -> String {
    if let Some(s) = name.strip_prefix('/') {
        s.to_string()
    } else {
        name
    }
}

fn from_hdf5_bytes(bytes: &[u8], span: Span) -> Result<Value> {
    let file = FileImage::new(bytes)?;
    to_record(&file, span)
}

pub fn signature() -> Signature {
    Signature::build("from hdf5")
        .usage("Convert from HDF5 binary into table")
        .allow_variants_without_examples(true)
        .input_output_types(vec![(Type::Binary, Type::Any)])
        .category(Category::Experimental)
        .filter()
}

pub fn run(call: &EvaluatedCall, input: &Value) -> Result<Value, LabeledError> {
    match input {
        Value::Binary { val, span } => from_hdf5_bytes(val, *span).map_err(|e| LabeledError {
            label: "HDF5 error".into(),
            msg: e.to_string(),
            span: Some(call.head),
        }),
        v => Err(LabeledError {
            label: "Expected binary from pipeline".into(),
            msg: format!("requires binary input, got {}", v.get_type()),
            span: Some(call.head),
        }),
    }
}
