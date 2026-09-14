mod conformer;
#[cfg(feature = "nemo-asr")]
mod external;
mod frontend;
mod projection;
mod subsampling;

pub use crate::asr::subsampling::SubsampledFeatures;
pub use conformer::{conformer_block_plan, conformer_encoder_plan};
#[cfg(feature = "nemo-asr")]
pub use external::NemotronAsr;
pub use frontend::NemotronFrontend;
pub use projection::pre_encode_projection;
pub use subsampling::{subsampling_plan, NemotronSubsampler};
