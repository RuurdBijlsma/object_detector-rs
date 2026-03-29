use crate::try2::try2;
use color_eyre::eyre::Result;

mod try1;
mod try2;

fn main() -> Result<()> {
    // try1()?;
    try2()?;

    Ok(())
}
