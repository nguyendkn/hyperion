//! Run-time loading of shared libraries (CUDA driver, cuBLAS), with no link-time dependency on
//! the CUDA toolkit: a build needs neither the toolkit nor a GPU, and a machine without the
//! library gets a typed error when a backend asks for it, never a crash at start-up.
//!
//! Only the three system calls that every platform has are declared here (`dlopen`/`dlsym` on
//! Unix, `LoadLibraryA`/`GetProcAddress` on Windows); no crate is involved.

use std::ffi::{c_char, c_void, CStr, CString};

/// Why a library or one of its functions could not be loaded.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LoadError {
    #[error("none of the libraries {tried:?} could be loaded: {reason}")]
    LibraryMissing { tried: Vec<String>, reason: String },
    #[error("function `{symbol}` was not found in {library}")]
    SymbolMissing { library: String, symbol: String },
}

#[cfg(unix)]
mod os {
    use std::ffi::{c_char, c_int, c_void};

    pub const RTLD_NOW: c_int = 2;

    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    extern "C" {
        pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        pub fn dlerror() -> *const c_char;
    }
}

#[cfg(windows)]
mod os {
    use std::ffi::{c_char, c_void};

    #[link(name = "kernel32")]
    extern "system" {
        pub fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        pub fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        pub fn GetLastError() -> u32;
    }
}

/// A loaded shared library. It is never unloaded: functions taken from it stay valid for the
/// life of the process, which is what lets callers hold plain function pointers.
pub struct Library {
    handle: *mut c_void,
    name: String,
}

// Safety: the handle is an opaque, immutable token for a library that stays loaded.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    /// Load the first of `names` that the system can find.
    pub fn open(names: &[&str]) -> Result<Self, LoadError> {
        let mut reason = String::from("no library names were given");
        for name in names {
            let Ok(c_name) = CString::new(*name) else { continue };
            match unsafe { open_one(&c_name) } {
                Ok(handle) => return Ok(Self { handle, name: (*name).to_string() }),
                Err(why) => reason = why,
            }
        }
        Err(LoadError::LibraryMissing { tried: names.iter().map(|n| (*n).to_string()).collect(), reason })
    }

    /// The function `name`, as a function pointer type `T`.
    ///
    /// # Safety
    /// `T` must be an `extern` function pointer type whose signature is exactly that of the
    /// library's function; calling it otherwise is undefined behaviour.
    pub unsafe fn symbol<T: Copy>(&self, name: &str) -> Result<T, LoadError> {
        debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
        let missing = || LoadError::SymbolMissing { library: self.name.clone(), symbol: name.to_string() };
        let c_name = CString::new(name).map_err(|_| missing())?;
        let address = find_symbol(self.handle, &c_name);
        if address.is_null() {
            return Err(missing());
        }
        Ok(std::mem::transmute_copy::<*mut c_void, T>(&address))
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(unix)]
unsafe fn open_one(name: &CStr) -> Result<*mut c_void, String> {
    let handle = os::dlopen(name.as_ptr(), os::RTLD_NOW);
    if handle.is_null() {
        let message = os::dlerror();
        Err(if message.is_null() { format!("{} could not be loaded", name.to_string_lossy()) } else { CStr::from_ptr(message).to_string_lossy().into_owned() })
    } else {
        Ok(handle)
    }
}

#[cfg(windows)]
unsafe fn open_one(name: &CStr) -> Result<*mut c_void, String> {
    let handle = os::LoadLibraryA(name.as_ptr());
    if handle.is_null() {
        Err(format!("{} could not be loaded (Windows error {})", name.to_string_lossy(), os::GetLastError()))
    } else {
        Ok(handle)
    }
}

#[cfg(unix)]
unsafe fn find_symbol(handle: *mut c_void, name: &CStr) -> *mut c_void {
    os::dlsym(handle, name.as_ptr() as *const c_char)
}

#[cfg(windows)]
unsafe fn find_symbol(handle: *mut c_void, name: &CStr) -> *mut c_void {
    os::GetProcAddress(handle, name.as_ptr() as *const c_char)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A library and a function every machine of the platform has.
    #[cfg(windows)]
    const SYSTEM: (&[&str], &str) = (&["kernel32.dll"], "GetTickCount64");
    #[cfg(all(unix, target_os = "linux"))]
    const SYSTEM: (&[&str], &str) = (&["libc.so.6"], "strlen");
    #[cfg(all(unix, not(target_os = "linux")))]
    const SYSTEM: (&[&str], &str) = (&["libSystem.B.dylib", "libc.so"], "strlen");

    #[test]
    fn a_missing_library_is_a_typed_error_listing_what_was_tried() {
        let error = Library::open(&["hyperion-no-such-library-1.so", "hyperion-no-such-library-2.dll"]).err().expect("must not load");
        match error {
            LoadError::LibraryMissing { tried, reason } => {
                assert_eq!(tried, vec!["hyperion-no-such-library-1.so", "hyperion-no-such-library-2.dll"]);
                assert!(!reason.is_empty());
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn the_first_library_that_loads_is_used_and_a_function_can_be_called() {
        let (names, function) = SYSTEM;
        let mut tried: Vec<&str> = vec!["hyperion-no-such-library.so"];
        tried.extend(names);
        let library = Library::open(&tried).expect("a system library must load");
        assert_eq!(library.name(), names[0], "the missing name is skipped, the next one is used");
        #[cfg(windows)]
        {
            let tick: unsafe extern "system" fn() -> u64 = unsafe { library.symbol(function).expect("symbol") };
            assert!(unsafe { tick() } > 0);
        }
        #[cfg(unix)]
        {
            let length: unsafe extern "C" fn(*const c_char) -> usize = unsafe { library.symbol(function).expect("symbol") };
            assert_eq!(unsafe { length(c"hello".as_ptr()) }, 5);
        }
    }

    #[test]
    fn a_missing_function_is_a_typed_error_naming_the_library_and_the_function() {
        let (names, _) = SYSTEM;
        let library = Library::open(names).expect("a system library must load");
        let error = unsafe { library.symbol::<*mut c_void>("hyperion_no_such_function") }.err().expect("must not resolve");
        match error {
            LoadError::SymbolMissing { library: lib, symbol } => {
                assert_eq!(lib, names[0]);
                assert_eq!(symbol, "hyperion_no_such_function");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }
}
