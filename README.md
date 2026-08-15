# tinyGPT

Personal implementation of Karpathy's [minGPT](https://github.com/karpathy/minGPT)

Aim of the project is to replicate a GPT-2 (124M) training run on a single 3090, but with some changes to the architecture and data.

## GPT 2 Training
For pre-training, we use a 10B token sample of the [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset, using the GPT2 tokeniser.

## SFT Training
After pre-training, we perform supervised fine tuning on the base model using the [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) dataset. Since the base GPT model only has a context size of 1024 tokens, the SFT dataset is reduced slightly be removing the system prompt where present. In cases where conversations are longer than the supported context window, the tokens are truncated to match the 1024 token limit (plus an appended EOS token).


## GPT 2 vs GPT 2 + SFT
The below shows some examples of responses to chat template prompts. We compare a base GPT 2 (124M) model with the finetuned model. You can see from the results that SFT does a lot better at producing conversational text in the chat format required! The actual outputs are still not great (not factual, not good code, some weird tangents) but this is expected for a model of only 124M parameters. But these initial results shows promise and is the model & data are scaled up then it should start to be much better!

---

**User Prompt:**
Hello how are you?

| Model | Full Token Sequence |
|---|---|
| **GPT-2** | &lt;&#124;user&#124;&gt;<br>Hello how are you?<br>&lt;&#124;assistant&#124;&gt;<br>I have my code and I am just going to write it.<br>&lt;&#124;&#124;password&gt; &lt;&#124;user&#124;&gt;<br>How does that match up?<br>&lt;User&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&gt; &lt;&#124;user&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;User&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;&#124;&lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt;&#124;user&#124;&gt; &lt; |
| **SFT GPT-2** | &lt;&#124;user&#124;&gt;<br>Hello how are you?<br>&lt;&#124;assistant&#124;&gt;<br>Hi there! I'm looking for a partner like you to help me find a job after a challenging internship. We're hoping to meet up after the summer holidays and explore different options. Our company has been searching for someone to work with and I’m curious about their company culture, values, and other considerations. Would you be interested in meeting up if there’s a time that suits you well? I look forward to making this connection! |

---

**User Prompt:**
Can you give me a recipe for a victoria sponge cake?

| Model | Full Token Sequence |
|---|---|
| **GPT-2** | &lt;&#124;user&#124;&gt;<br>Can you give me a recipe for a victoria sponge cake?<br>&lt;&#124;assistant&#124;&gt;<br>Hi! Nice one!<br>This is really delicious and we all do so so very well.<br>And the best frosting would be a cake with blue raspberry and chocolate.<br>Thanks for sharing!<br>Looks so lovely!<br>&lt;li&gt;I'm very happy you think of a frosting cake. I think there are many great recipes out there. Maybe I should start making a similar cake using some strawberry raspberry. I think you're going to love it as much as I do.<br>But, the cake won't be the same.<br>I love you.<br>What is your favorite cake recipe?<br>&lt;li&gt;I'm so happy I have come to know that I actually made a strawberry cake. It turned out pretty delicious as well.<br>Now that this cake is made, what are you waiting for?<br>&#124;behalf&#124;&gt;How cool would this be for a blueberry cake recipe?&#124;&#124;i&#124;<br>&#124;behalf&#124;&gt;How pretty would this be for a pin |
| **SFT GPT-2** | &lt;&#124;user&#124;&gt;<br>Can you give me a recipe for a victoria sponge cake?<br>&lt;&#124;assistant&#124;&gt;<br>Certainly! Here's one of my favorite recipes on the internet:<br>A moist, creamy frosting<br>3 cups sugar<br>4 eggs<br>1 cup instant coffee<br>2 cups white chocolate<br>2 cups unsalted butter<br>1 teaspoon vanilla extract<br>1 cup heavy cream<br>1 teaspoon powdered sugar<br>Sugar for sprinkling<br>In a large bowl, beat the sugar until well combined. Beat the eggs, egg yolks, and vanilla extract until smooth and set in a greased 9x13 pan. Pour the batter into the frosting and pour the frosting over the remaining frosting. Enjoy! |
