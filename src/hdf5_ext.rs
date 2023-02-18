use core::ffi::{c_size_t, c_uint, c_void};
use hdf5::{h5call, types::TypeDescriptor, Dataset, Datatype, File, Result};
use hdf5_sys::{h5d::H5Dread, h5i::hid_t, h5p::H5P_DEFAULT, h5s::H5S_ALL};
use std::{marker::PhantomData, ops::Deref};

const H5LT_FILE_IMAGE_DONT_COPY: c_uint = 0x2;
const H5LT_FILE_IMAGE_DONT_RELEASE: c_uint = 0x4;

#[link(name = "hdf5_hl")]
extern "C" {
    fn H5LTopen_file_image(buf_ptr: *mut c_void, buf_size: c_size_t, flags: c_uint) -> hid_t;
}

#[derive(Clone)]
pub struct FileImage<'a> {
    file: File,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> FileImage<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let hid = h5call!(H5LTopen_file_image(
            bytes.as_ptr() as *const c_void as _,
            bytes.len(),
            H5LT_FILE_IMAGE_DONT_COPY | H5LT_FILE_IMAGE_DONT_RELEASE
        ))?;
        let file: File = unsafe { std::mem::transmute(hid) };
        Ok(Self {
            file,
            _p: PhantomData,
        })
    }
}

impl Deref for FileImage<'_> {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.file
    }
}

pub trait ReadRawBytes {
    fn read_raw_bytes(&self, dtype: &TypeDescriptor) -> Result<Vec<u8>>;
}

impl ReadRawBytes for Dataset {
    fn read_raw_bytes(&self, dtype: &TypeDescriptor) -> Result<Vec<u8>> {
        let len = self.size();
        let item_size = dtype.size();
        let mut buffer = Vec::with_capacity(len * item_size);
        // Convert again to fit the current native endian.
        let native_dtype = Datatype::from_descriptor(dtype)?;
        h5call!(H5Dread(
            self.id(),
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
}
