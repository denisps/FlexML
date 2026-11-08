use std::ffi::CStr;
use std::{
    cell::RefCell,
    io::Write,
    marker::PhantomData,
    ptr::NonNull,
    sync::{
        Once,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Debug)]
pub struct Error(&'static str);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LogLevel {
    Off,
    Warn,
    Debug,
}

const LOG_OFF: usize = usize::MAX;
static LOG_LEVEL: AtomicUsize = AtomicUsize::new(LOG_OFF);
static LOG_CALLBACK_INIT: Once = Once::new();

/// Sets the process-global GGML log filter without locking on each log call.
///
/// Call this before creating a [`Graph`] or otherwise causing GGML to log. The
/// callback is installed once; later calls only update an atomic filter level.
pub fn set_log_level(level: LogLevel) {
    let threshold = match level {
        LogLevel::Off => LOG_OFF,
        LogLevel::Warn => ggml_sys::ggml_log_level_GGML_LOG_LEVEL_WARN as usize,
        LogLevel::Debug => ggml_sys::ggml_log_level_GGML_LOG_LEVEL_NONE as usize,
    };
    LOG_LEVEL.store(threshold, Ordering::Release);
    LOG_CALLBACK_INIT.call_once(|| unsafe {
        ggml_sys::ggml_log_set(
            Some(ggml_log_callback),
            (&LOG_LEVEL as *const AtomicUsize).cast_mut().cast(),
        );
    });
}

unsafe extern "C" fn ggml_log_callback(
    level: ggml_sys::ggml_log_level,
    text: *const std::ffi::c_char,
    user_data: *mut std::ffi::c_void,
) {
    if user_data.is_null() || text.is_null() {
        return;
    }
    let threshold = unsafe { &*user_data.cast::<AtomicUsize>() }.load(Ordering::Acquire);
    let level = level as usize;
    let is_continuation = level == ggml_sys::ggml_log_level_GGML_LOG_LEVEL_CONT as usize;
    if threshold == LOG_OFF || (!is_continuation && level < threshold) {
        return;
    }

    let text = unsafe { CStr::from_ptr(text) }.to_bytes();
    let mut stderr = std::io::stderr().lock();
    let _ = stderr.write_all(text);
    let _ = stderr.flush();
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Cpu,
    Metal,
    OpenCl,
    Vulkan,
    Cuda,
}

impl Backend {
    /// Returns the backend selected by this crate's Cargo features.
    pub const fn selected() -> Self {
        #[cfg(feature = "metal")]
        {
            return Self::Metal;
        }
        #[cfg(feature = "opencl")]
        {
            return Self::OpenCl;
        }
        #[cfg(feature = "vulkan")]
        {
            return Self::Vulkan;
        }
        #[cfg(feature = "cuda")]
        {
            return Self::Cuda;
        }
        #[allow(unreachable_code)]
        Self::Cpu
    }
}

pub struct Graph {
    ctx: NonNull<ggml_sys::ggml_context>,
    backend: NonNull<ggml_sys::ggml_backend>,
    buffer: RefCell<Option<NonNull<ggml_sys::ggml_backend_buffer>>>,
    inputs: RefCell<Vec<(NonNull<ggml_sys::ggml_tensor>, f32)>>,
    _not_send: PhantomData<*mut ()>,
}

pub struct Tensor<'g> {
    raw: NonNull<ggml_sys::ggml_tensor>,
    _graph: PhantomData<&'g Graph>,
}

impl Graph {
    pub fn new() -> Result<Self, Error> {
        let ctx = unsafe {
            ggml_sys::ggml_init(ggml_sys::ggml_init_params {
                mem_size: 16 * 1024 * 1024,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            })
        };
        let ctx = NonNull::new(ctx).ok_or(Error("ggml_init failed"))?;
        let backend = match create_backend() {
            Some(backend) => backend,
            None => {
                unsafe { ggml_sys::ggml_free(ctx.as_ptr()) };
                return Err(Error("failed to initialize the selected GGML backend"));
            }
        };
        Ok(Self {
            ctx,
            backend,
            buffer: RefCell::new(None),
            inputs: RefCell::new(Vec::new()),
            _not_send: PhantomData,
        })
    }

    pub const fn backend(&self) -> Backend {
        Backend::selected()
    }

    pub fn scalar(&self, value: f32) -> Result<Tensor<'_>, Error> {
        if self.buffer.borrow().is_some() {
            return Err(Error(
                "cannot add tensors after graph computation has started",
            ));
        }
        let raw = unsafe {
            ggml_sys::ggml_new_tensor_1d(self.ctx.as_ptr(), ggml_sys::ggml_type_GGML_TYPE_F32, 1)
        };
        let raw = NonNull::new(raw).ok_or(Error("failed to create tensor"))?;
        self.inputs.borrow_mut().push((raw, value));
        Ok(Tensor {
            raw,
            _graph: PhantomData,
        })
    }

    pub fn add<'g>(&'g self, a: &Tensor<'g>, b: &Tensor<'g>) -> Result<Tensor<'g>, Error> {
        self.binary(a, b, true)
    }

    pub fn mul<'g>(&'g self, a: &Tensor<'g>, b: &Tensor<'g>) -> Result<Tensor<'g>, Error> {
        self.binary(a, b, false)
    }

    fn binary<'g>(
        &'g self,
        a: &Tensor<'g>,
        b: &Tensor<'g>,
        add: bool,
    ) -> Result<Tensor<'g>, Error> {
        if self.buffer.borrow().is_some() {
            return Err(Error(
                "cannot add operations after graph computation has started",
            ));
        }
        let raw = unsafe {
            if add {
                ggml_sys::ggml_add(self.ctx.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr())
            } else {
                ggml_sys::ggml_mul(self.ctx.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr())
            }
        };
        Ok(Tensor {
            raw: NonNull::new(raw).ok_or(Error("failed to create operation"))?,
            _graph: PhantomData,
        })
    }

    pub fn compute<'g>(&'g self, result: &Tensor<'g>) -> Result<f32, Error> {
        let graph = unsafe { ggml_sys::ggml_new_graph(self.ctx.as_ptr()) };
        let graph = NonNull::new(graph).ok_or(Error("failed to create computation graph"))?;
        unsafe { ggml_sys::ggml_build_forward_expand(graph.as_ptr(), result.raw.as_ptr()) };

        if self.buffer.borrow().is_none() {
            let buffer = unsafe {
                ggml_sys::ggml_backend_alloc_ctx_tensors(self.ctx.as_ptr(), self.backend.as_ptr())
            };
            let buffer = NonNull::new(buffer).ok_or(Error("failed to allocate backend tensors"))?;
            *self.buffer.borrow_mut() = Some(buffer);

            for (tensor, value) in self.inputs.borrow().iter() {
                unsafe {
                    ggml_sys::ggml_backend_tensor_set(
                        tensor.as_ptr(),
                        (value as *const f32).cast(),
                        0,
                        std::mem::size_of::<f32>(),
                    );
                }
            }
        }

        let status =
            unsafe { ggml_sys::ggml_backend_graph_compute(self.backend.as_ptr(), graph.as_ptr()) };
        if status != ggml_sys::ggml_status_GGML_STATUS_SUCCESS {
            return Err(Error("GGML backend graph computation failed"));
        }

        let mut value = 0.0_f32;
        unsafe {
            ggml_sys::ggml_backend_tensor_get(
                result.raw.as_ptr(),
                (&mut value as *mut f32).cast(),
                0,
                std::mem::size_of::<f32>(),
            );
        }
        Ok(value)
    }
}

#[cfg(feature = "metal")]
fn create_backend() -> Option<NonNull<ggml_sys::ggml_backend>> {
    NonNull::new(unsafe { ggml_sys::ggml_backend_metal_init() })
}

#[cfg(feature = "opencl")]
fn create_backend() -> Option<NonNull<ggml_sys::ggml_backend>> {
    NonNull::new(unsafe { ggml_sys::ggml_backend_opencl_init() })
}

#[cfg(feature = "cuda")]
fn create_backend() -> Option<NonNull<ggml_sys::ggml_backend>> {
    NonNull::new(unsafe { ggml_sys::ggml_backend_cuda_init(0) })
}

#[cfg(feature = "vulkan")]
fn create_backend() -> Option<NonNull<ggml_sys::ggml_backend>> {
    use std::ffi::CString;
    let name = CString::new("Vulkan").ok()?;
    let registry = unsafe { ggml_sys::ggml_backend_reg_by_name(name.as_ptr()) };
    let registry = NonNull::new(registry)?;
    let device = unsafe { ggml_sys::ggml_backend_reg_dev_get(registry.as_ptr(), 0) };
    let device = NonNull::new(device)?;
    NonNull::new(unsafe { ggml_sys::ggml_backend_dev_init(device.as_ptr(), std::ptr::null()) })
}

#[cfg(not(any(
    feature = "metal",
    feature = "opencl",
    feature = "vulkan",
    feature = "cuda"
)))]
fn create_backend() -> Option<NonNull<ggml_sys::ggml_backend>> {
    NonNull::new(unsafe { ggml_sys::ggml_backend_cpu_init() })
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            if let Some(buffer) = self.buffer.get_mut().take() {
                ggml_sys::ggml_backend_buffer_free(buffer.as_ptr());
            }
            ggml_sys::ggml_backend_free(self.backend.as_ptr());
            ggml_sys::ggml_free(self.ctx.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, Graph};

    #[test]
    fn selected_backend_computes_a_graph() {
        super::set_log_level(super::LogLevel::Off);
        let graph = Graph::new().expect("initialize selected backend");
        assert_eq!(graph.backend(), Backend::selected());
        let two = graph.scalar(2.0).unwrap();
        let three = graph.scalar(3.0).unwrap();
        let five = graph.add(&two, &three).unwrap();
        let result = graph.mul(&five, &three).unwrap();
        assert_eq!(graph.compute(&result).unwrap(), 15.0);
    }
}
