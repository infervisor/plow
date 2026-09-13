pub struct LayerNames {
    ff1_norm_weight: String,
    ff1_norm_bias: String,
    ff1_expand: String,
    ff1_contract: String,
    attention_norm_weight: String,
    attention_norm_bias: String,
    query: String,
    key: String,
    value: String,
    position: String,
    attention_output: String,
    bias_u: String,
    bias_v: String,
    convolution_norm_weight: String,
    convolution_norm_bias: String,
    pointwise_in: String,
    depthwise: String,
    channel_norm_weight: String,
    channel_norm_bias: String,
    pointwise_out: String,
    ff2_norm_weight: String,
    ff2_norm_bias: String,
    ff2_expand: String,
    ff2_contract: String,
    output_norm_weight: String,
    output_norm_bias: String,
}

pub struct SubsamplingNames {
    weight: String,
    bias: String,
}

impl SubsamplingNames {
    pub fn new(index: usize) -> Self {
        let prefix = format!("encoder.pre_encode.conv.{index}");
        Self {
            weight: format!("{prefix}.weight"),
            bias: format!("{prefix}.bias"),
        }
    }

    pub fn weight(&self) -> &str {
        &self.weight
    }

    pub fn bias(&self) -> &str {
        &self.bias
    }
}

impl LayerNames {
    pub fn new(layer: usize) -> Self {
        let prefix = format!("encoder.layers.{layer}");
        Self {
            ff1_norm_weight: format!("{prefix}.norm_feed_forward1.weight"),
            ff1_norm_bias: format!("{prefix}.norm_feed_forward1.bias"),
            ff1_expand: format!("{prefix}.feed_forward1.linear1.weight"),
            ff1_contract: format!("{prefix}.feed_forward1.linear2.weight"),
            attention_norm_weight: format!("{prefix}.norm_self_att.weight"),
            attention_norm_bias: format!("{prefix}.norm_self_att.bias"),
            query: format!("{prefix}.self_attn.linear_q.weight"),
            key: format!("{prefix}.self_attn.linear_k.weight"),
            value: format!("{prefix}.self_attn.linear_v.weight"),
            position: format!("{prefix}.self_attn.linear_pos.weight"),
            attention_output: format!("{prefix}.self_attn.linear_out.weight"),
            bias_u: format!("{prefix}.self_attn.pos_bias_u"),
            bias_v: format!("{prefix}.self_attn.pos_bias_v"),
            convolution_norm_weight: format!("{prefix}.norm_conv.weight"),
            convolution_norm_bias: format!("{prefix}.norm_conv.bias"),
            pointwise_in: format!("{prefix}.conv.pointwise_conv1.weight"),
            depthwise: format!("{prefix}.conv.depthwise_conv.weight"),
            channel_norm_weight: format!("{prefix}.conv.batch_norm.weight"),
            channel_norm_bias: format!("{prefix}.conv.batch_norm.bias"),
            pointwise_out: format!("{prefix}.conv.pointwise_conv2.weight"),
            ff2_norm_weight: format!("{prefix}.norm_feed_forward2.weight"),
            ff2_norm_bias: format!("{prefix}.norm_feed_forward2.bias"),
            ff2_expand: format!("{prefix}.feed_forward2.linear1.weight"),
            ff2_contract: format!("{prefix}.feed_forward2.linear2.weight"),
            output_norm_weight: format!("{prefix}.norm_out.weight"),
            output_norm_bias: format!("{prefix}.norm_out.bias"),
        }
    }

    pub fn borrow(&self) -> devgen::conformer::ConformerLayerWeights<'_> {
        use devgen::conformer::{
            AttentionWeights, ConformerLayerWeights, ConvolutionWeights, FeedForwardWeights,
        };
        ConformerLayerWeights {
            feed_forward1: FeedForwardWeights {
                norm: norm(&self.ff1_norm_weight, &self.ff1_norm_bias),
                expand: &self.ff1_expand,
                contract: &self.ff1_contract,
            },
            attention: AttentionWeights {
                norm: norm(&self.attention_norm_weight, &self.attention_norm_bias),
                query: &self.query,
                key: &self.key,
                value: &self.value,
                position: &self.position,
                output: &self.attention_output,
                bias_u: &self.bias_u,
                bias_v: &self.bias_v,
            },
            convolution: ConvolutionWeights {
                norm: norm(&self.convolution_norm_weight, &self.convolution_norm_bias),
                pointwise_in: &self.pointwise_in,
                depthwise: &self.depthwise,
                channel_norm: norm(&self.channel_norm_weight, &self.channel_norm_bias),
                pointwise_out: &self.pointwise_out,
            },
            feed_forward2: FeedForwardWeights {
                norm: norm(&self.ff2_norm_weight, &self.ff2_norm_bias),
                expand: &self.ff2_expand,
                contract: &self.ff2_contract,
            },
            output_norm: norm(&self.output_norm_weight, &self.output_norm_bias),
        }
    }
}

fn norm<'a>(weight: &'a str, bias: &'a str) -> devgen::conformer::NormWeights<'a> {
    devgen::conformer::NormWeights {
        gamma: weight,
        beta: bias,
    }
}
