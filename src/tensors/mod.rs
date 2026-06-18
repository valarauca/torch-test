use std::{
    path::Path,
    rc::{Rc},
    ops::Deref,
};
use tch::{Kind};
use mmap_guard::{FileData,map_file};
use anamnesis::parse::safetensors::{
    SafetensorsHeader,parse_safetensors_header,Dtype,
};

/// Wrapper around safe tensor data
#[derive(Clone)]
pub struct SafeTensor {
    // rc is intentional, once we have multi-gigabyte safe tensor loaded
    // and we're tealing with cuda drivers we can't have stuff moving between
    // threads willy-nilly.
    rc: Rc<InnerSafeTensor>,
}
struct InnerSafeTensor {
    guard: FileData,
    header: SafetensorsHeader,
}
impl SafeTensor {

    /// Load a safe tensor file
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let guard = map_file(path)?;
        let header = parse_safetensors_header(&guard)?;       
        let t = InnerSafeTensor { header, guard };
        Ok(Self { rc: Rc::new(t) })
    }

    /// Returns the tch `Kind` for a named tensor, or `None` if not present.
    pub fn kind_of(&self, name: &str) -> Option<tch::Kind> {
        let dtype = self.rc.header.tensors.iter()
            .find(|e| e.name.as_str() == name)?
            .dtype;
        match dtype {
            Dtype::BF16   => Some(tch::Kind::BFloat16),
            Dtype::F16    => Some(tch::Kind::Half),
            Dtype::F32    => Some(tch::Kind::Float),
            Dtype::F64    => Some(tch::Kind::Double),
            Dtype::Bool   => Some(tch::Kind::Bool),
            Dtype::U8     => Some(tch::Kind::Uint8),
            Dtype::I8     => Some(tch::Kind::Int8),
            Dtype::I16    => Some(tch::Kind::Int16),
            Dtype::I32    => Some(tch::Kind::Int),
            Dtype::I64    => Some(tch::Kind::Int64),
            Dtype::F8E5M2 => Some(tch::Kind::Float8e5m2),
            _             => None,
        }
    }

    /// returns tensor names
    pub fn names<'a>(&'a self) -> impl Iterator<Item=&'a str> {
        self.rc.header.tensors.iter().map(|entry| entry.name.as_str())
    }

    /// Returns a "wrapped" a tensor.
    ///
    /// The type will dereference to a `tch::Tensor`. This is treated as "special" as the tensor
    /// cannot be resized due to it sharing a memory map of the underlying file.
    pub fn get_tensor(&self, name: &str, kind: tch::Kind, device: tch::Device) -> anyhow::Result<Option<TensorWrapper>> {
        let entry = match self.rc.header.tensors.iter().filter(|entry| entry.name.as_str() == name).next() {
            None => return Ok(None),
            Some(e) => e,
        };
        match (entry.dtype,kind) {
            (Dtype::BF16,Kind::BFloat16) |
            (Dtype::F16,Kind::Half) |
            (Dtype::F32,Kind::Float) |
            (Dtype::F64,Kind::Double) |
            (Dtype::Bool,Kind::Bool) |
            (Dtype::U8,Kind::Uint8) |
            (Dtype::I8,Kind::Int8) |
            (Dtype::I16,Kind::Int16) |
            (Dtype::I32,Kind::Int) |
            (Dtype::I64,Kind::Int64)|
            (Dtype::F8E5M2,Kind::Float8e5m2) => { },
            (input,request) => {
                anyhow::bail!("cannot tensor '{}' requested '{:?}' file contains '{:?}'", name, request, input);
            }
        };
        // re-allocate shape & strides
        let shape = entry.shape.iter().map(|&d|d as i64).collect::<Vec<i64>>();
        let mut strides = vec![1i64; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as i64;
        }

        // data_offsets are relative to the data section which begins at header_size + 8
        let base = self.rc.header.header_size + 8;
        let r = (base + entry.data_offsets.0)..(base + entry.data_offsets.1);
        let data = &self.rc.deref().guard[r];

        // construct the tensor in torch
        // SAFETY: three invariants hold here.
        // 1. Pointer lifetime: TensorWrapper retains a SafeTensor clone (Rc<InnerSafeTensor>),
        //    keeping the FileData mmap alive for the full lifetime of the tensor.
        // 2. No mutation: FileData::Mapped wraps memmap2::Mmap (PROT_READ only), so any
        //    in-place PyTorch op through Deref will SIGSEGV rather than silently corrupt the map.
        // 3. Alignment: safetensors pads every tensor's data region to 8-byte boundaries,
        //    covering all dtypes matched above (max alignment requirement: 8 bytes for F64/I64).
        let tensor = unsafe {
            tch::Tensor::f_from_blob(
                data.as_ptr(),
                &shape,
                &strides,
                kind,
                device,
            )?
        };
        Ok(Some(TensorWrapper {
            // keep a reference to the memory map around
            map: self.clone(),
            tensor,
        }))
    }
}

/// Uses `rc` as once a tensor is loaded (potentially) into GPU memory we shoudln't be passing it
/// between threads
pub struct TensorWrapper {
    map: SafeTensor,
    tensor: tch::Tensor,
}
impl std::ops::Deref for TensorWrapper {
    type Target = tch::Tensor;
    fn deref(&self) -> &Self::Target { &self.tensor }
}
