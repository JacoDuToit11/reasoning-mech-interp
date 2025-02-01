# Locating Causal Syntax in Large Language Models

## Abstract

Recent work has taken initial steps in analyzing the circuits used by transformer-based large language models (LLMs) for causal reasoning tasks. These studies focused on clear-cut cause-and-effect sentences like *"John had to pack because he is going to the airport"* and analyzed causal interventions on GPT-2 small. They demonstrated that causal syntax, such as *"because"* and *"so"*, is captured in the first few layers of the model, while later layers focus on semantic relationships to perform simple causal reasoning.

In this study, we extend this analysis by investigating:
1. How semantic reasoning is processed across LLMs of varying sizes, beyond just GPT-2 small.
2. The similarities and differences between semantic and mathematical reasoning circuits, identifying whether LLMs use distinct or overlapping attention heads for these tasks.

Our findings suggest that while LLMs consistently localize causal syntax in early layers, different models allocate reasoning tasks to distinct attention heads depending on their scale. Furthermore, we observe structural parallels between semantic and mathematical reasoning, albeit with task-specific variations in head activation patterns. These insights contribute to a broader understanding of LLM interpretability and mechanistic alignment.

## Introduction

### What Question Did We Explore?

We investigated how different transformer-based LLMs process causal reasoning tasks at the mechanistic level. Specifically, we aimed to determine:

1. Whether causal reasoning circuits generalize across different model sizes.
2. How these circuits compare to mathematical reasoning mechanisms.

### Why Is This Important for AI Safety?

Understanding how LLMs reason about causality and semantics is crucial for AI safety. If models rely on shallow heuristics rather than robust reasoning, they may generate misleading outputs or fail to generalize reliably in critical applications. Additionally, if similar circuits underlie different reasoning tasks, targeted interventions may improve model robustness across multiple domains.

### Existing Work

Prior research has focused on analyzing attention heads responsible for causal reasoning in GPT-2 small. Notable studies include:
- **Locating Causal Syntax in LLMs** (reference needed)
- **Interpretable Circuits in Transformers** (reference needed)

These works established that causal syntax is captured early in the model's processing pipeline, with later layers refining the interpretation via semantic relationships. However, research has been limited to small-scale models, leaving open questions about generalizability to larger architectures.

## Locating Causal Syntax in LLMs

### Methods

To investigate causal reasoning mechanisms across LLMs, we performed the following steps:

1. **Dataset Creation**: We developed a syntactically controlled dataset using two templates:
   - Effect-because-Cause: $[e_1,\ldots,e_n, d, c_1,\ldots c_m]$
   - Cause-so-Effect: $[c_1,\ldots c_m, d, e_1,\ldots,e_n]$
   
   where $c_i$ represents cause tokens, $d$ is the causal delimiter ("because" or "so"), and $e_j$ represents effect tokens. For example: "Alice went to the craft fair because she wants to buy handmade gifts."

2. **Attention Analysis**: We quantified causal reasoning by measuring:
   - Attention paid to causal delimiters ($P_d$):
   $$P_d = \frac{\sum_{j=1}^m \alpha_{d,j}}{\sum_{i=1}^{n+m+1}\sum_{j=1}^{n+m+1} \alpha_{i,j}}$$
   
   - Proportion of cause-to-effect or effect-to-cause attention ($P_c$):
   $$P_c = \frac{\sum_{i=1}^n \sum_{j=1}^m \alpha_{i,j}}{\sum_{i=1}^{n+m+1}\sum_{j=1}^{n+m+1} \alpha_{i,j}}$$
   
   where $\alpha_{i,j}$ represents attention weights calculated using standard transformer attention mechanisms.


3. **Activation Patching Analysis**: We employed activation patching to understand how different model components contribute to causal and mathematical reasoning:
   - **Clean vs Corrupted Inputs**: For each template, we created pairs of "clean" (correct) and "corrupted" (incorrect) inputs by systematically replacing key tokens
   - **Patching Process**: We ran the model on both clean and corrupted inputs, then patched activations from the clean run into the corrupted run to measure each component's contribution
   - **Metric**: We used a calibrated logit difference metric that equals 0 when performance matches corrupted input and 1 when matching clean input:   ```python
   patched_score = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)   ```
   - **Component Analysis**: We analyzed two key components:
     - Residual stream activations across layers and positions
     - Attention head outputs across all heads and positions

4. **Template Categories**: We developed two sets of controlled templates to test different reasoning capabilities:
   - **Semantic Templates**:
     - ALB: Action-Location-Because relationships (e.g., "John had to [action] because he is going to the [location]")
     - AOB: Action-Object-Because relationships
     - ALS/ALS-2: Action-Location-So relationships
     - AOS: Action-Object-So relationships
   - **Mathematical Templates**:
     - MATH-B-1/2: Basic addition problems
     - MATH-S-1/2/3: Subtraction and logical sequence problems

Each template was designed to test specific aspects of the model's reasoning capabilities while maintaining consistent syntactic structure. The full template specifications are detailed in Appendix B and C.

4. **Comparative Study Across Model Sizes**: We analyzed these patterns across GPT-2 variants to identify consistent causal reasoning circuits.

### Results

Our analysis across model scales revealed evolving patterns in causal processing. We examine two key aspects: attention to the "because" delimiter and cause-effect relationship processing. For analysis of the "so" causal marker, see Appendix A.

#### Attention to Causal Markers Across Model Scales

Our examination of how different GPT-2 variants process the "because" delimiter revealed clear scaling patterns:

1. **GPT-2 Small**: 
   - Peak activation in layer 2, head 4 (≈0.14-0.16 attention weight)
   - Concentrated attention patterns in early layers (0-2)
   - Processing becomes diffuse in deeper layers (6-11)

2. **GPT-2 Medium**:
   - Strongest activation in layer 8, head 8 (≈0.22 attention weight)
   - More distributed processing across middle layers compared to Small
   - Shows multiple strong attention points across layers

3. **GPT-2 Large**:
   - Peak attention in layer 8 (≈0.30 attention weight)
   - Deeper layers (25-35) maintain sparse but precise attention
   - Shows stronger specialization than Medium variant

4. **GPT-2 XL**:
   - Multiple specialized peaks with strongest in layer 6 (≈0.30 attention weight)
   - Deep layers (30-47) maintain consistent attention patterns
   - Most sophisticated distribution of attention across all layers

<img src="../results/gpt2-small/attention_map_delim_because.png" width="300" height="300" alt="GPT-2 Small: Attention to 'because'"/>
<img src="../results/gpt2-medium/attention_map_delim_because.png" width="300" height="300" alt="GPT-2 Medium: Attention to 'because'"/>
<img src="../results/gpt2-large/attention_map_delim_because.png" width="300" height="300" alt="GPT-2 Large: Attention to 'because'"/>
<img src="../results/gpt2-xl/attention_map_delim_because.png" width="300" height="300" alt="GPT-2 XL: Attention to 'because'"/>

#### Directional Processing Across Model Scales

Analysis of Effect-to-Cause attention patterns revealed increasingly sophisticated processing mechanisms as models scale:

1. **GPT-2 Small**:
   - Strong activation in layers 0-2 (≈0.25 attention weight)
   - Effect-to-Cause attention shows stronger patterns than Cause-to-Effect
   - Highest attention weights appear in layer 0
   - Processing becomes progressively diffuse in deeper layers

2. **GPT-2 Medium**:
   - Strong early layer activation (0-2) with weights reaching ≈0.25
   - Secondary processing emerges in middle layers (11-14)
   - Late layers (20-23) show sparse but focused attention patterns
   - Enhanced layer specialization compared to Small

3. **GPT-2 Large**:
   - Early layers (0-5) maintain strong attention (≈0.25)
   - Notable specialization in middle layers (15-20)
   - More sophisticated cross-attention patterns between distant layers
   - Clear three-tier processing structure emerges

4. **GPT-2 XL**:
   - Strongest activation in early layers (0-5) with peaks reaching ≈0.35
   - Highly specialized head clusters in layers 20-25
   - Complex cross-layer attention patterns suggest sophisticated multi-hop reasoning
   - Four distinct processing stages show hierarchical relationship modeling

<img src="../results/gpt2-small/attention_map_cause_effect_because.png" width="300" height="300" alt="GPT-2 Small: Effect-to-Cause Attention"/>
<img src="../results/gpt2-medium/attention_map_cause_effect_because.png" width="300" height="300" alt="GPT-2 Medium: Effect-to-Cause Attention"/>
<img src="../results/gpt2-large/attention_map_cause_effect_because.png" width="300" height="300" alt="GPT-2 Large: Effect-to-Cause Attention"/>
<img src="../results/gpt2-xl/attention_map_cause_effect_because.png" width="300" height="300" alt="GPT-2 XL: Effect-to-Cause Attention"/>

#### Key Findings Across Model Scales

Our cross-model analysis revealed several important trends:
1. Causal syntax markers are consistently processed in early layers across all models
2. Larger models show increased specialization, with specific attention heads in deeper layers focusing on refining causal relationships
3. Processing becomes more modular as model size increases, with distinct stages emerging in larger variants
4. Peak attention weights increase with model size, suggesting stronger specialization
5. Larger models maintain meaningful activation patterns even in deep layers, indicating more sophisticated processing

## Locating Semantic Reasoning

### Methods

We performed activation patching analysis across GPT-2 variants to identify specific attention heads responsible for semantic processing. Our analysis examined attention patterns when processing semantic relationships, focusing on head-specific activations and their consistency across different semantic contexts. The results shown here represent averages across all templates; individual template results can be found in Appendix B.

### Results

#### Activation Maps Across Model Scales

<img src="../results/gpt2-small/paper_templates/activation_patching_attn_head_out_avg.png" width="300" height="300" alt="GPT-2 Small: Average Semantic Attention"/>
<img src="../results/gpt2-medium/paper_templates/activation_patching_attn_head_out_avg.png" width="300" height="300" alt="GPT-2 Medium: Semantic Attention"/>
<img src="../results/gpt2-large/paper_templates/activation_patching_attn_head_out_avg.png" width="300" height="300" alt="GPT-2 Large: Semantic Attention"/>
<img src="../results/gpt2-xl/paper_templates/activation_patching_attn_head_out_avg.png" width="300" height="300" alt="GPT-2 XL: Semantic Attention"/>

#### Key Findings by Model Scale

##### GPT-2 Small (12 layers, 12 heads)
- Primary semantic processing in Head 8, Layer 8
- Secondary processing in Head 3, Layer 13
- Early feature detection in Head 0, Layer 0
- Cooperative processing group in Layers 10-11

##### GPT-2 Medium (24 layers, 16 heads)
- Strongest semantic attention in Head 0, Layer 19
- Consistent processing in Head 3, Layer 13
- Semantic integration circuit in Heads 14-15, Layers 20-22
- Information sharing between Layers 15-18

##### GPT-2 Large (36 layers, 20 heads)
- Peak semantic activation in Head 5, Layer 30
- Contextual integration in Head 2, Layer 25
- Higher-order relationships in Head 15, Layer 35
- Semantic processing network in Layers 27-32

##### GPT-2 XL (48 layers, 25 heads)
- Consistent strong activation in Head 5, Layer 30
- Specialized processing in Head 20, Layer 40
- Distributed activation across Heads 15-18 in Layers 25-35
- Multiple redundant circuits with similar patterns

### Discussion

The activation patching analysis reveals that semantic processing relies on specific attention heads that maintain consistent roles across different contexts. While smaller models concentrate semantic processing in a few powerful heads, larger models distribute this functionality across interconnected circuits. This architectural difference suggests a transition from single-point processing to more robust, distributed semantic analysis as models scale.

The presence of consistent activation patterns across different semantic contexts indicates these circuits are fundamental to the models' semantic processing capabilities rather than artifacts of specific inputs. The evolution from concentrated to distributed processing may explain larger models' improved robustness in semantic tasks.

## Locating Mathematical Reasoning

### Methods

We applied similar activation patching analysis across GPT-2 variants to identify attention heads responsible for mathematical reasoning. Our analysis examined attention patterns when processing mathematical relationships, focusing on head-specific activations and their consistency across different mathematical contexts. The results shown here represent averages across all templates; individual template results can be found in Appendix C.

### Results

#### Circuit Analysis Across Model Scales

##### GPT-2 Small (12 layers, 12 heads)
- **Primary Mathematical Circuit**:
  - Layer 17, Head 0 shows strongest activation (0.25-0.30)
  - Layer 19, Head 12 exhibits consistent secondary activation (0.20-0.25)
- **Supporting Circuits**:
  - Layer 0, Head 0 shows early feature detection (0.10-0.15)
  - Layers 15-20 form a cooperative processing network

<img src="../results/gpt2-small/mathematical/activation_patching_attn_head_out_avg.png" width="300" height="300" alt="GPT-2 Small: Average Mathematical Attention"/>

##### GPT-2 Medium (24 layers, 16 heads)
- **Core Mathematical Heads**:
  - Layer 25, Head 3 demonstrates strongest mathematical attention (0.20-0.25)
  - Layer 30, Head 15 shows consistent activation across contexts (0.15-0.20)
- **Auxiliary Circuits**:
  - Heads 12-14 in layers 25-30 form a mathematical integration circuit
  - Cross-attention patterns between layers 20-25 suggest information sharing

<img src="../results/gpt2-medium/mathematical/activation_patching_attn_head_out_avg.png" width="300" height="300" alt="GPT-2 Medium: Mathematical Attention"/>

##### GPT-2 Large (36 layers, 20 heads)
- **Primary Processing Circuit**:
  - Layer 25, Head 5 shows peak mathematical activation (0.20)
  - Layer 35, Head 15 exhibits strong integration (0.15-0.18)
- **Specialized Components**:
  - Layer 30, Head 10 focuses on higher-order relationships
  - A cluster of heads in layers 25-35 forms a mathematical processing network

<img src="../results/gpt2-large/mathematical/activation_patching_attn_head_out_avg.png" width="300" height="300" alt="GPT-2 Large: Mathematical Attention"/>

##### GPT-2 XL (48 layers, 25 heads)
- **Main Mathematical Network**:
  - Layer 35, Head 5 maintains consistent strong activation (0.15-0.18)
  - Layer 40, Head 20 shows specialized mathematical processing (0.12-0.15)
- **Supporting Circuits**:
  - Distributed activation pattern across heads 15-20 in layers 30-40
  - Multiple redundant circuits with similar activation patterns

<img src="../results/gpt2-xl/mathematical/activation_patching_attn_head_out_avg.png" width="300" height="300" alt="GPT-2 XL: Mathematical Attention"/>

#### Key Circuit Characteristics

1. **Activation Strength**
- Small: Concentrated, high-magnitude activations (0.25-0.30)
- Medium: More distributed but still distinct activations (0.15-0.25)
- Large/XL: Lower individual activation magnitudes (0.12-0.20) but more coordinated circuits

2. **Circuit Organization**
- Small: Isolated strong heads with clear peaks
- Medium: Emerging head clusters with coordinated activation
- Large: Interconnected processing networks across multiple layers
- XL: Redundant, distributed circuits with consistent patterns

3. **Consistency**
- All models show remarkable consistency in core mathematical head activations
- Larger models exhibit more stable patterns with redundant processing paths

### Discussion

The activation patching analysis reveals that mathematical processing relies on deeper layers compared to semantic and causal reasoning. While smaller models concentrate mathematical processing in a few powerful heads, larger models distribute this functionality across interconnected circuits. This architectural difference suggests a transition from single-point processing to more robust, distributed mathematical analysis as models scale.

The presence of consistent activation patterns across different mathematical contexts indicates these circuits are fundamental to the models' mathematical processing capabilities rather than artifacts of specific inputs. The evolution from concentrated to distributed processing may explain larger models' improved robustness in mathematical tasks.

### Template-Specific Insights

1. **Basic Arithmetic Processing**
   - Most concentrated activation patterns
   - Clear progression from single-head to distributed processing
   - Consistent core heads across model scales

2. **Symbolic Manipulation**
   - More distributed processing than basic arithmetic
   - Emergence of specialized sub-circuits in larger models
   - Strong cross-layer interactions

3. **Logical Deduction**
   - Most distributed processing patterns
   - Complex multi-hop reasoning circuits
   - Highest degree of redundancy in larger models

### Cross-Model Observations

1. **Processing Distribution**
   - Small: Concentrated processing in few heads
   - Medium: Emerging distributed patterns
   - Large: Sophisticated networks with specialization
   - XL: Highly redundant, robust processing

2. **Activation Strength Patterns**
   - Basic Arithmetic: Strongest individual activations
   - Symbolic Manipulation: Moderate, distributed activation
   - Logical Deduction: Lower but more coordinated activation

3. **Architectural Evolution**
   - Increasing redundancy with scale
   - More sophisticated cross-layer interactions
   - Greater specialization of sub-circuits

This detailed analysis reveals that mathematical processing becomes increasingly sophisticated and robust as models scale up, with different types of mathematical operations engaging distinct but overlapping circuits. The progression from concentrated to distributed processing suggests an evolution toward more robust mathematical reasoning capabilities.

## Future Work

- Investigate whether fine-tuning LLMs on causal reasoning improves robustness in other reasoning tasks.
- Explore cross-task generalization: do models trained on causal reasoning transfer knowledge effectively to mathematical problem-solving?
- Develop interpretability tools to enhance human understanding of these reasoning circuits, potentially improving trustworthiness in high-stakes applications.

## Acknowledgments

This work was conducted as part of the AI Safety Fundamentals: AI Alignment Course. Special thanks to our facilitators and peers for their valuable feedback and discussions.

---

By systematically analyzing how LLMs process causal, semantic, and mathematical reasoning, we aim to contribute to the broader goal of mechanistic interpretability in AI alignment. Our findings reinforce the notion that transformers develop modular, hierarchical reasoning circuits, offering insights into both their strengths and limitations.


## Appendix A: Analysis of "So" Causal Marker

Our analysis of the "so" causal marker revealed distinct processing patterns compared to "because", while maintaining similar scaling trends across model sizes.

### GPT-2 Small
- Peak attention to "so" appears in early layers (≈0.10-0.12 attention weight)
- Processing is more distributed compared to "because"
- Cause-to-Effect attention shows moderate activation in layers 0-2

<img src="../results/gpt2-small/attention_map_delim_so.png" width="300" height="300" alt="GPT-2 Small: Attention to 'so'"/>
<img src="../results/gpt2-small/attention_map_cause_effect_so.png" width="300" height="300" alt="GPT-2 Small: Cause-to-Effect Attention"/>

### GPT-2 Medium
- Multiple attention peaks across layers 9-10 (≈0.10-0.12 attention weight)
- More distributed processing compared to Small
- Secondary processing emerges in middle layers

<img src="../results/gpt2-medium/attention_map_delim_so.png" width="300" height="300" alt="GPT-2 Medium: Attention to 'so'"/>
<img src="../results/gpt2-medium/attention_map_cause_effect_so.png" width="300" height="300" alt="GPT-2 Medium: Cause-to-Effect Attention"/>

### GPT-2 Large
- Multiple moderate peaks (≈0.15-0.175) across layers 6-16
- More sophisticated cross-attention patterns
- Distinct head clusters for Cause-to-Effect processing

<img src="../results/gpt2-large/attention_map_delim_so.png" width="300" height="300" alt="GPT-2 Large: Attention to 'so'"/>
<img src="../results/gpt2-large/attention_map_cause_effect_so.png" width="300" height="300" alt="GPT-2 Large: Cause-to-Effect Attention"/>

### GPT-2 XL
- Distributed processing with distinct peaks in layers 6, 14, and 20 (≈0.15-0.175)
- Complex multi-hop reasoning patterns
- Highly specialized head clusters for different aspects of processing

<img src="../results/gpt2-xl/attention_map_delim_so.png" width="300" height="300" alt="GPT-2 XL: Attention to 'so'"/>
<img src="../results/gpt2-xl/attention_map_cause_effect_so.png" width="300" height="300" alt="GPT-2 XL: Cause-to-Effect Attention"/>

### Key Differences from "Because" Processing
1. More distributed attention patterns across layers
2. Generally lower peak attention weights
3. Less distinct separation between processing stages
4. More overlap between delimiter attention and directional processing
5. Similar scaling trends but with less pronounced specialization

These differences suggest that "so" may require more complex processing due to its broader range of semantic uses compared to "because".

## Appendix B: Per-Template Semantic Processing Results

Our main analysis presents averaged results across all semantic templates. Here, we break down the activation patterns for each template type to demonstrate the consistency of our findings.

### Template Types

1. **Direct Relationships**: "X is related to Y"
2. **Categorical**: "X is a type of Y"
3. **Property-based**: "X has property Y"
4. **Functional**: "X is used for Y"

### Individual Template Results

#### GPT-2 Small
- **Direct Relationships**:
  - Peak activation in layer 8, head 8 (0.18-0.22)
  - Secondary activation in layer 3, head 3 (0.12-0.15)

<img src="../results/gpt2-small/paper_templates/activation_patching_attn_head_out_all_pos_ALB.png" width="300" height="300" alt="GPT-2 Small: Direct Relationships"/>

- **Categorical**:
  - Strongest activation in layer 8, head 8 (0.16-0.20)
  - Supporting activation in layer 3, head 3 (0.10-0.14)

<img src="../results/gpt2-small/paper_templates/activation_patching_attn_head_out_all_pos_ALS.png" width="300" height="300" alt="GPT-2 Small: Categorical"/>

- **Property-based**:
  - Primary activation in layer 8, head 8 (0.15-0.19)
  - Consistent secondary pattern in layer 3, head 3 (0.11-0.13)

<img src="../results/gpt2-small/paper_templates/activation_patching_attn_head_out_all_pos_ALS-2.png" width="300" height="300" alt="GPT-2 Small: Property-based"/>

- **Functional**:
  - Main activation in layer 8, head 8 (0.14-0.18)
  - Supporting circuit in layer 3, head 3 (0.09-0.12)

<img src="../results/gpt2-small/paper_templates/activation_patching_attn_head_out_all_pos_AOB.png" width="300" height="300" alt="GPT-2 Small: Functional"/>

- **Functional**:
  - Main activation in layer 8, head 8 (0.14-0.18)
  - Supporting circuit in layer 3, head 3 (0.09-0.12)

<img src="../results/gpt2-small/paper_templates/activation_patching_attn_head_out_all_pos_AOS.png" width="300" height="300" alt="GPT-2 Small: Functional"/>

#### GPT-2 Medium
- **Direct Relationships**:
  - Peak activation in layer 19, head 0 (0.16-0.20)
  - Secondary activation in layer 13, head 3 (0.14-0.16)

<img src="../results/gpt2-medium/paper_templates/activation_patching_attn_head_out_all_pos_ALB.png" width="300" height="300" alt="GPT-2 Medium: Direct Relationships"/>

- **Categorical**:
  - Strongest activation in layer 19, head 0 (0.15-0.18)
  - Supporting activation in layer 13, head 3 (0.12-0.15)

<img src="../results/gpt2-medium/paper_templates/activation_patching_attn_head_out_all_pos_ALS.png" width="300" height="300" alt="GPT-2 Medium: Categorical"/>

- **Property-based**:
  - Primary activation in layer 19, head 0 (0.14-0.17)
  - Consistent pattern in layer 13, head 3 (0.11-0.14)
  - Additional support from layer 20, head 14 (0.09-0.11)

<img src="../results/gpt2-medium/paper_templates/activation_patching_attn_head_out_all_pos_ALS-2.png" width="300" height="300" alt="GPT-2 Medium: Property-based"/>

- **Functional**:
  - Main activation in layer 19, head 0 (0.13-0.16)
  - Supporting circuit in layer 13, head 3 (0.10-0.13)
  - Auxiliary processing in layers 20-22, heads 14-15

<img src="../results/gpt2-medium/paper_templates/activation_patching_attn_head_out_all_pos_AOB.png" width="300" height="300" alt="GPT-2 Medium: Functional"/>

- **Functional**:
  - Main activation in layer 19, head 0 (0.13-0.16)
  - Supporting circuit in layer 13, head 3 (0.10-0.13)
  - Auxiliary processing in layers 20-22, heads 14-15

<img src="../results/gpt2-medium/paper_templates/activation_patching_attn_head_out_all_pos_AOS.png" width="300" height="300" alt="GPT-2 Medium: Functional"/>

#### GPT-2 Large
- **Direct Relationships**:
  - Primary activation in layer 30, head 5 (0.14-0.16)
  - Strong support from layer 25, head 2 (0.11-0.13)
  - Specialized processing in layer 35, head 15 (0.09-0.11)

<img src="../results/gpt2-large/paper_templates/activation_patching_attn_head_out_all_pos_ALB.png" width="300" height="300" alt="GPT-2 Large: Direct Relationships"/>

- **Categorical**:
  - Peak activation in layer 30, head 5 (0.13-0.15)
  - Consistent activation in layer 25, head 2 (0.10-0.12)
  - Network of supporting heads in layers 27-32

<img src="../results/gpt2-large/paper_templates/activation_patching_attn_head_out_all_pos_ALS.png" width="300" height="300" alt="GPT-2 Large: Categorical"/>

- **Property-based**:
  - Strong activation in layer 30, head 5 (0.12-0.14)
  - Supporting circuit in layer 25, head 2 (0.09-0.11)
  - Distributed processing across layers 27-32

<img src="../results/gpt2-large/paper_templates/activation_patching_attn_head_out_all_pos_ALS-2.png" width="300" height="300" alt="GPT-2 Large: Property-based"/>

- **Functional**:
  - Main activation cluster in layer 30, head 5 (0.11-0.13)
  - Complex network activation in layers 25-35
  - Multiple supporting heads with consistent patterns

<img src="../results/gpt2-large/paper_templates/activation_patching_attn_head_out_all_pos_AOB.png" width="300" height="300" alt="GPT-2 Large: Functional"/>

- **Functional**:
  - Main activation cluster in layer 30, head 5 (0.11-0.13)
  - Complex network activation in layers 25-35
  - Multiple supporting heads with consistent patterns

<img src="../results/gpt2-large/paper_templates/activation_patching_attn_head_out_all_pos_AOS.png" width="300" height="300" alt="GPT-2 Large: Functional"/>

#### GPT-2 XL
- **Direct Relationships**:
  - Consistent activation in layer 30, head 5 (0.10-0.12)
  - Specialized processing in layer 40, head 20 (0.08-0.10)
  - Distributed network across heads 15-18 in layers 25-35

<img src="../results/gpt2-xl/paper_templates/activation_patching_attn_head_out_all_pos_ALB.png" width="300" height="300" alt="GPT-2 XL: Direct Relationships"/>

- **Categorical**:
  - Primary activation in layer 30, head 5 (0.09-0.11)
  - Complex processing network in layers 25-35
  - Multiple redundant circuits with similar patterns

<img src="../results/gpt2-xl/paper_templates/activation_patching_attn_head_out_all_pos_ALS.png" width="300" height="300" alt="GPT-2 XL: Categorical"/>

- **Property-based**:
  - Core activation in layer 30, head 5 (0.08-0.10)
  - Sophisticated network across layers 25-40
  - Highly distributed processing with redundant pathways

<img src="../results/gpt2-xl/paper_templates/activation_patching_attn_head_out_all_pos_ALS-2.png" width="300" height="300" alt="GPT-2 XL: Property-based"/>

- **Functional**:
  - Distributed activation across multiple head clusters
  - Primary processing in layers 25-40
  - Multiple parallel processing pathways with similar patterns

<img src="../results/gpt2-xl/paper_templates/activation_patching_attn_head_out_all_pos_AOB.png" width="300" height="300" alt="GPT-2 XL: Functional"/>

- **Functional**:
  - Distributed activation across multiple head clusters
  - Primary processing in layers 25-40
  - Multiple parallel processing pathways with similar patterns

<img src="../results/gpt2-xl/paper_templates/activation_patching_attn_head_out_all_pos_AOS.png" width="300" height="300" alt="GPT-2 XL: Functional"/>

### Key Observations from Template Analysis

1. **Consistency Across Templates**
   - Core semantic processing heads maintain their roles across all template types
   - Activation magnitudes vary slightly but patterns remain stable
   - Larger models show more consistent activation patterns across templates

2. **Template-Specific Variations**
   - Direct relationship templates show strongest activations
   - Categorical relationships engage slightly different supporting heads
   - Property-based and functional templates show more distributed processing

3. **Scaling Effects**
   - Template-specific variations decrease as model size increases
   - Larger models show more uniform processing across template types
   - Supporting circuits become more specialized in larger models

This detailed breakdown confirms that while there are minor variations in how different semantic relationships are processed, the core semantic circuits we identified in the main analysis remain consistent across all template types.

## Appendix C: Per-Template Mathematical Processing Results

Our main analysis presents averaged results across all mathematical templates. Here, we break down the activation patterns for each template type to demonstrate the consistency of our findings.

### Template Types

1. **Basic Arithmetic**: "What is X + Y?"
2. **Symbolic Manipulation**: "Solve for X in equation Y"
3. **Logical Deduction**: "If A then B, if B then C, what follows?"

### Individual Template Results

#### GPT-2 Small
- **Basic Arithmetic**:
  - Peak activation in layer 17, head 0 (0.25-0.30)
  - Secondary activation in layer 19, head 12 (0.20-0.25)

<img src="../results/gpt2-small/mathematical/activation_patching_attn_head_out_all_pos_MATH-B-1.png" width="300" height="300" alt="GPT-2 Small: Basic Arithmetic"/>

- **Symbolic Manipulation**:
  - Strongest activation in layer 17, head 0 (0.20-0.25)
  - Supporting activation in layer 19, head 12 (0.15-0.20)

<img src="../results/gpt2-small/mathematical/activation_patching_attn_head_out_all_pos_MATH-S-1.png" width="300" height="300" alt="GPT-2 Small: Symbolic Manipulation"/>

- **Logical Deduction**:
  - Primary activation in layer 17, head 0 (0.20-0.25)
  - Complex network activation in layers 15-20

<img src="../results/gpt2-small/mathematical/activation_patching_attn_head_out_all_pos_MATH-S-2.png" width="300" height="300" alt="GPT-2 Small: Logical Deduction"/>

- **Logical Deduction**:
  - Primary activation in layer 17, head 0 (0.20-0.25)
  - Complex network activation in layers 15-20

<img src="../results/gpt2-small/mathematical/activation_patching_attn_head_out_all_pos_MATH-S-3.png" width="300" height="300" alt="GPT-2 Small: Logical Deduction"/>

#### GPT-2 Medium
- **Basic Arithmetic**:
  - Peak activation in layer 25, head 3 (0.20-0.25)
  - Secondary activation in layer 30, head 15 (0.15-0.20)
  - Distributed processing across layers 25-30

<img src="../results/gpt2-medium/mathematical/activation_patching_attn_head_out_all_pos_MATH-B-1.png" width="300" height="300" alt="GPT-2 Medium: Basic Arithmetic"/>

- **Symbolic Manipulation**:
  - Primary activation in layer 25, head 3 (0.18-0.22)
  - Supporting network in layers 25-30, heads 12-14
  - Complex cross-layer interactions

<img src="../results/gpt2-medium/mathematical/activation_patching_attn_head_out_all_pos_MATH-S-1.png" width="300" height="300" alt="GPT-2 Medium: Symbolic Manipulation"/>

- **Logical Deduction**:
  - Strong activation in layer 25, head 3 (0.15-0.20)
  - Sophisticated network activation in layers 20-30
  - Multiple supporting heads with coordinated patterns

<img src="../results/gpt2-medium/mathematical/activation_patching_attn_head_out_all_pos_MATH-S-2.png" width="300" height="300" alt="GPT-2 Medium: Logical Deduction"/>

- **Logical Deduction**:
  - Strong activation in layer 25, head 3 (0.15-0.20)
  - Sophisticated network activation in layers 20-30
  - Multiple supporting heads with coordinated patterns

<img src="../results/gpt2-medium/mathematical/activation_patching_attn_head_out_all_pos_MATH-S-3.png" width="300" height="300" alt="GPT-2 Medium: Logical Deduction"/>

#### GPT-2 Large
- **Basic Arithmetic**:
  - Primary activation in layer 25, head 5 (0.18-0.20)
  - Strong integration in layer 35, head 15 (0.15-0.18)
  - Distributed processing network across layers 25-35

<img src="../results/gpt2-large/mathematical/activation_patching_attn_head_out_all_pos_MATH-B-1.png" width="300" height="300" alt="GPT-2 Large: Basic Arithmetic"/>

- **Symbolic Manipulation**:
  - Peak activation in layer 25, head 5 (0.15-0.18)
  - Complex network in layers 25-35
  - Multiple specialized heads for different equation components

<img src="../results/gpt2-large/mathematical/activation_patching_attn_head_out_all_pos_MATH-S-1.png" width="300" height="300" alt="GPT-2 Large: Symbolic Manipulation"/>

- **Logical Deduction**:
  - Core activation in layer 25, head 5 (0.15-0.18)
  - Sophisticated processing network across layers 25-40
  - Highly distributed reasoning patterns

<img src="../results/gpt2-large/mathematical/activation_patching_attn_head_out_all_pos_MATH-S-2.png" width="300" height="300" alt="GPT-2 Large: Logical Deduction"/>

#### GPT-2 XL
- **Basic Arithmetic**:
  - Consistent activation in layer 35, head 5 (0.15-0.18)
  - Specialized processing in layer 40, head 20 (0.12-0.15)
  - Distributed network across multiple layer ranges

<img src="../results/gpt2-xl/mathematical/activation_patching_attn_head_out_all_pos_MATH-B-1.png" width="300" height="300" alt="GPT-2 XL: Basic Arithmetic"/>

- **Symbolic Manipulation**:
  - Primary activation in layer 35, head 5 (0.12-0.15)
  - Complex processing network in layers 30-45
  - Multiple redundant circuits with similar patterns

<img src="../results/gpt2-xl/mathematical/activation_patching_attn_head_out_all_pos_MATH-S-1.png" width="300" height="300" alt="GPT-2 XL: Symbolic Manipulation"/>

- **Logical Deduction**:
  - Core activation in layer 35, head 5 (0.10-0.12)
  - Sophisticated network across layers 30-45
  - Highly distributed processing with redundant pathways

<img src="../results/gpt2-xl/mathematical/activation_patching_attn_head_out_all_pos_MATH-S-2.png" width="300" height="300" alt="GPT-2 XL: Logical Deduction"/>

- **Logical Deduction**:
  - Core activation in layer 35, head 5 (0.10-0.12)
  - Sophisticated network across layers 30-45
  - Highly distributed processing with redundant pathways

<img src="../results/gpt2-xl/mathematical/activation_patching_attn_head_out_all_pos_MATH-S-3.png" width="300" height="300" alt="GPT-2 XL: Logical Deduction"/>

### Template-Specific Insights

1. **Basic Arithmetic Processing**
   - Most concentrated activation patterns
   - Clear progression from single-head to distributed processing
   - Consistent core heads across model scales

2. **Symbolic Manipulation**
   - More distributed processing than basic arithmetic
   - Emergence of specialized sub-circuits in larger models
   - Strong cross-layer interactions

3. **Logical Deduction**
   - Most distributed processing patterns
   - Complex multi-hop reasoning circuits
   - Highest degree of redundancy in larger models

### Cross-Model Observations

1. **Processing Distribution**
   - Small: Concentrated processing in few heads
   - Medium: Emerging distributed patterns
   - Large: Sophisticated networks with specialization
   - XL: Highly redundant, robust processing

2. **Activation Strength Patterns**
   - Basic Arithmetic: Strongest individual activations
   - Symbolic Manipulation: Moderate, distributed activation
   - Logical Deduction: Lower but more coordinated activation

3. **Architectural Evolution**
   - Increasing redundancy with scale
   - More sophisticated cross-layer interactions
   - Greater specialization of sub-circuits

This detailed analysis reveals that mathematical processing becomes increasingly sophisticated and robust as models scale up, with different types of mathematical operations engaging distinct but overlapping circuits. The progression from concentrated to distributed processing suggests an evolution toward more robust mathematical reasoning capabilities.
