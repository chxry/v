#![feature(proc_macro_diagnostic, vec_into_raw_parts)]
use std::process::Command;
use proc_macro::TokenStream;
use syn::{parse_macro_input, LitStr};

#[proc_macro]
pub fn include_glsl(item: TokenStream) -> TokenStream {
  let s = parse_macro_input!(item as LitStr);
  match Command::new("glslc")
    .arg(s.value())
    .args(["-o", "-"])
    .output()
  {
    Ok(out) => {
      if out.status.success() {
        let v = out.stdout.into_raw_parts();
        format!("{:?}", unsafe {
          Vec::from_raw_parts(v.0 as *mut u32, v.1 / 4, v.2 / 4)
        })
        .parse()
        .unwrap()
      } else {
        s.span()
          .unwrap()
          .error("couldnt compile")
          .note(String::from_utf8_lossy(&out.stderr))
          .emit();
        "[0]".parse().unwrap()
      }
    }
    Err(e) => {
      s.span()
        .unwrap()
        .error("could not run glslc")
        .note(e.to_string())
        .emit();
      "[0]".parse().unwrap()
    }
  }
}
