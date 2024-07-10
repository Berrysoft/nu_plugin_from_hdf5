use crate::hdf5_ext::{FileImage, ReadRawBytes};
use hdf5::{
    types::{FloatSize, IntSize, TypeDescriptor, VarLenArray, VarLenAscii, VarLenUnicode},
    Dataset, Group, Result,
};
use nu_protocol::{Category, LabeledError, PipelineData, Record, Signature, Span, Type, Value};

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
            internal_span: $span,
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
            Value::Record {
                val: Record::from_raw_cols_vals(cols, vals, span, span)
                    .unwrap()
                    .into(),
                internal_span: span,
            }
        }
        TypeDescriptor::FixedArray(ty, len) => {
            assert_eq!(slice.len(), ty.size() * len);
            Value::List {
                vals: slice
                    .chunks(ty.size())
                    .map(|slice| to_value(slice, ty, span))
                    .try_collect()?,
                internal_span: span,
            }
        }
        TypeDescriptor::FixedAscii(len) | TypeDescriptor::FixedUnicode(len) => {
            assert_eq!(slice.len(), *len);
            Value::String {
                val: String::from_utf8_lossy(slice).into_owned(),
                internal_span: span,
            }
        }
        TypeDescriptor::VarLenArray(ty) => {
            let hvl = unsafe { native!(VarLenArray<u8>, slice).as_ref() }.unwrap();
            Value::List {
                vals: hvl
                    .chunks(ty.size())
                    .map(|slice| to_value(slice, ty, span))
                    .try_collect()?,
                internal_span: span,
            }
        }
        TypeDescriptor::VarLenAscii => {
            let str = unsafe { native!(VarLenAscii, slice).as_ref() }.unwrap();
            Value::String {
                val: str.as_str().to_string(),
                internal_span: span,
            }
        }
        TypeDescriptor::VarLenUnicode => {
            let str = unsafe { native!(VarLenUnicode, slice).as_ref() }.unwrap();
            Value::String {
                val: str.as_str().to_string(),
                internal_span: span,
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
    Ok(Value::List {
        vals,
        internal_span: span,
    })
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
    Ok(Value::Record {
        val: Record::from_raw_cols_vals(cols, vals, span, span)
            .unwrap()
            .into(),
        internal_span: span,
    })
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

pub fn run(input: PipelineData) -> Result<PipelineData, LabeledError> {
    match input {
        PipelineData::Empty => Ok(PipelineData::Empty),
        PipelineData::Value(v, meta) => match v {
            Value::Binary {
                val,
                internal_span: span,
            } => {
                let value =
                    from_hdf5_bytes(&val, span).map_err(|e| LabeledError::new(e.to_string()))?;
                Ok(PipelineData::Value(value, meta))
            }
            v => Err(LabeledError::new(format!(
                "requires binary input, got {}",
                v.get_type()
            ))),
        },
        PipelineData::ListStream(_, _) => Err(LabeledError::new("unsupported list stream")),
        PipelineData::ByteStream(stream, meta) => {
            let value = stream.into_value()?;
            match value {
                Value::Binary {
                    val,
                    internal_span: span,
                } => {
                    let value = from_hdf5_bytes(&val, span)
                        .map_err(|e| LabeledError::new(e.to_string()))?;
                    Ok(PipelineData::Value(value, meta))
                }
                _ => unreachable!(),
            }
        }
    }
}
