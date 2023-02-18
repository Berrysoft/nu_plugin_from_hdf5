use core::ffi::c_void;
use hdf5::{
    h5call,
    types::{FloatSize, IntSize, TypeDescriptor, VarLenArray, VarLenAscii, VarLenUnicode},
    Dataset, Datatype, File, Group, Result,
};
use hdf5_sys::{h5d::H5Dread, h5p::H5P_DEFAULT, h5s::H5S_ALL};
use nu_protocol::{Span, Value};

const H5LT_FILE_IMAGE_DONT_COPY: u32 = 0x2;
const H5LT_FILE_IMAGE_DONT_RELEASE: u32 = 0x4;

#[link(name = "hdf5_hl")]
extern "C" {
    fn H5LTopen_file_image(buf_ptr: *mut c_void, buf_size: usize, flags: u32) -> i64;
}

fn read_raw_vec(dataset: &Dataset, dtype: &TypeDescriptor) -> Result<Vec<u8>> {
    let len = dataset.size();
    let item_size = dtype.size();
    let mut buffer = Vec::with_capacity(len * item_size);
    // Convert again to fit the current native endian.
    let native_dtype = Datatype::from_descriptor(dtype)?;
    h5call!(H5Dread(
        dataset.id(),
        native_dtype.id(),
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        buffer.spare_capacity_mut().as_mut_ptr() as *mut _
    ))?;
    unsafe {
        buffer.set_len(len * item_size);
    }
    Ok(buffer)
}

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
    let data = read_raw_vec(dataset, &dtype)?;
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

pub fn from_hdf5_bytes(bytes: &[u8], span: Span) -> Result<Value> {
    let hid = h5call!(H5LTopen_file_image(
        bytes.as_ptr() as *const c_void as _,
        bytes.len(),
        H5LT_FILE_IMAGE_DONT_COPY | H5LT_FILE_IMAGE_DONT_RELEASE
    ))?;
    let file: File = unsafe { std::mem::transmute(hid) };
    to_record(&file, span)
}
