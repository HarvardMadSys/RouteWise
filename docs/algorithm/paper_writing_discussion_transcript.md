Yeah, since others have not joined, maybe we can wait a few minutes before we start to talk about draft.
So have you made the newest change?
I never finished a change on Overleaf.
Okay. Do you want to commit the changes you have already made?
If something you have already changed, we don't have to discuss it.
Yeah.
And also, if you use Dropbox, so when we move to interactive changes,
you probably need to get a manual lock, because otherwise if you push a change,
like if you make a change in Dropbox, it basically overwrites the whole file.
Yeah.
Let's say if someone makes changes and Dropbox upload a new file, the changes will be overwritten.
Yeah. Okay. Maybe we should get started.
Let's go for the draft.
Should we wait for the end?
We can wait for the end, but we also get started.
Let me ask.
Let me go through my modification first.
Okay.
First, in the experiment, I have some analysis about sensitivity about the predict lens of output token.
It does not matter as we discussed previously.
And add some details of experiment also in the panics part.
This is the first modification.
The second one.
Oh, by the way, we probably don't want to use, yeah, we can talk about that later.
And in online.
When I walk through the paper, I realize the proof here, the sketch here is a bit wrong.
And I, instead of the proof here by just sight the serum itself, I never prove it because it's a mature result, right?
This is the point to, okay, I haven't checked this part.
I will take a look later.
We probably won't have proof in the paper.
There could be more panics if we had proof.
Yeah, okay.
So just leave the serum here and move the proof part to a panics.
Yeah, but you probably need to place the citation if you don't have a proof sketch.
Yeah.
And I also update the hedging part, including the experiment and the new results.
That's basically what I did so far.
Okay.
Okay, great.
Yeah, I think on the writing part.
Right now the paper is, it just needs more work.
So we need some structuring, a restructuring, and also then we might need more experiments, especially our ablation study.
Right now we don't have any experiments.
When we start from the beginning, we can read through the paper together.
Yeah, so I can explain what I think.
Because I started writing comments, I realized it may not be clear and I need to write a lot.
It's easier to just go through it together.
Let me share my screen.
So when we write the introduction last, like when we have all other sections,
because the instruction is basically a mini background, motivation, design, and evaluation.
Design and evaluation is kind of minimal, but it's more like a mini background and motivation.
So that's why we write it at last.
So we don't write it at the beginning because every time you change the later part, you are changing the introduction.
It's the same for abstract because abstract is kind of somewhere off everything.
Yeah, one thing we need to change as you mentioned yesterday is we need to merge the linearity and the cost.
So right now the cost and linearity is kind of separated.
Well, it's okay to separately discuss thoughts and linearity as two steps.
But we need a system that considers both.
So yeah, we can talk about the last. Let's skip intro.
So these two figures, we probably don't have this large figures.
We probably will merge them into a single column.
And we merge them into a single column.
I think the font size right now is actually okay. It's not too small.
Let me see if we make it a single column.
Is this already?
Yeah, this is already right.
Yeah, I think we're going into single column walks.
We probably also want to add the small parameters to show there isn't this spiky.
Like spiky latency.
That depends on whether there is spiky or not.
It could be a bug and they are not that spiky.
But if we have, since we have more players and more profiling results, we can serve the animal.
I think we can just use different ways to present it.
So specific on this figure, let's see.
We probably don't want to global penalize.
Okay, I think this is fine probably.
But before, like sub figures, it is better to plot each figure as one figure,
but then like combine them into one figure.
Because this allows us to change the figure easily.
Like whether you change to top and bottom or change it to like in the same role.
And also when we describe the figure, we can use sub figure to like to refer to it.
Okay, let's come back to intonator.
So I think we need a background section.
We can make changes while we talk about it.
Or more freely you want to make changes.
Very easy for you to just make changes.
So we need a background section to talk about our influence.
Like the unique characteristics of our influence.
For example, non-linear application cost with sequence sense and what else.
And also the bottleneck is not only the compute.
It can be memory bandwidth and memory capacity.
So there are multiple bottlenecks in our solving.
Yeah, and maybe others.
We need to have a background section to talk about some background and things we may need later.
So let's see what else we need.
Let me kind of think about what we need to add to background.
Taxonomy and problem formulation.
We can also talk about different pricing model.
Yeah, pricing models can go to the background.
So basically these three different types of providers.
But we don't want the formulas to go to the background.
So in the background you basically talk about there are three types of providers.
And give an example of how their pricing model works.
And just using like plain language, not using formula.
But we also talk about that remembering the introduction part.
So we should imagine them twice in two consecutive.
Yeah, it's okay.
I mean, in the intro, we all have overlap with everything.
It's kind of a mini, like when people read through the intro,
they will have a whole idea of what you do.
Yeah, intro should be self-contained.
You won't want to give very detailed design in the intro.
Or how you solve the problem in the intro.
You will give a high level overview of the idea.
So basically for intro, you will have some paragraphs on background.
Like I'm solving some background on the provider, like the hybrid pricing.
Those are all good.
This is a little bit abrupt.
I'm still not sure how we integrated this into it.
We need to think about it.
Yeah, this is too much.
I mean, this is too much.
Oh, and the paragraph title is too much into detail.
We don't need to give it so much detail in intro.
The idea you talk about in this paragraph, we certainly need some of it.
Because we need to say that the quota-based provider,
if you calculate per token price, is cheaper.
That's kind of an opportunity to cost.
And we should know which is a quota-based provider when we have capacity.
This is the kind of idea how you use a quota-based provider
and a capacity-based provider to lower costs.
And this is how latency comes in.
It can be a step on top of the cost.
When you introduce other providers, we say their latency may not be good.
Sometimes not good.
So that's why we need to improve latency.
We also need to reduce latency.
So if you'd like to just choose a quota-based provider and a capacity-based provider,
if you just like to prioritize them, you will get really bad latency.
That's something we need to show.
So it should not be two paragraphs.
Just like naturally after introduced the cost part
and then mentioned the latency could be another optimization.
When you say that latency is a problem first,
for intro, you always have the background, the current state,
and the problem and why it needs to be solved and how you solve it.
And our structure, yeah.
Don't worry about latency for now.
We can come back later.
Let's finish the rest.
Yeah, so for contribution, I think we probably won't write in this way.
The taxonomy definition, yeah, this is more like a machine learning paper.
So we probably would, we talk about first the system.
We identify the different pricing models
and designed algorithm to leverage quota and the kind of cost-based provider team
to optimize for cost.
Then we also, similarly, we talk about latency.
When we talk about the system you built,
then you talk about the evaluation.
It's kind of for contribution.
And so should I write still like this bullet point format?
We have bullet points.
But I mean, you can have kind of a title here, but it's not necessary.
The contribution needs to be shot.
Right now it's maybe now purple point.
Try to make it like two, three, nine, and not five, six, or eight lines.
Yeah, and also just focus on high level.
The algorithm is just one contribution.
We don't need to separate them into multiple.
The problem definition, we don't need that.
It can be part of one of the contributions.
Yeah, we will come back to this when we have the later sections.
Because right now it reflects the later sections.
That's why you wrote it in this way.
When the later sections change, this will also change.
Yeah.
So the taxonomy and problem formulation will become part of the design.
So after background, we need to talk about the motivation.
So in background, you are talking about different pricing model.
There you will also talk a little bit about motivation,
how we leverage the pricing model, the high level idea.
It's similar to your intro.
Don't worry about redundancy.
People need some redundancy.
Because otherwise, if people miss one sentence, they will get lost.
But we certainly copy one paragraph from the intro here.
The different parts will have different focus.
But usually we will have redundancy.
So once we have the background,
if one last subsection talks about the pricing model and motivation,
then we will have design.
So design needs to have a high level overview first.
So that's basically the components in the system.
That's basically how your quest flows in the system.
So you will need to have an overview section at the beginning.
What does the system look like?
You don't need to go into details about like database or like logging or like a dashboard.
It's also not needed.
The components that's needed for the system to work.
For example, you will need a router.
And inside the router, you will need an algorithm that decides between different providers.
You can draw like some kind of smart boring or something like that.
But they should show this algorithm decides the providers for each request.
You can also say you use private two along that to say this is our algorithm that decides this.
Then there's also a component which optimizes for latency.
Those all need to be in the system diagram.
So basically when people read that diagram, they need to understand how the system works.
They may not know how the private two algorithm works.
Because they know there's an algorithm that decides the provider for each request.
And then after that step, there's another algorithm that decides like among the few chosen providers,
or sorted providers, there's another algorithm that looks at the latency profile
and to choose a provider based on the latency profile to minimize latency.
Then you also have smart hedging.
So those all need to be in the diagram.
So basically when people read the diagram, they know how it works.
In the overview, you will talk about the components, different components in the system.
You don't need to go into detail on how they work.
For example, how the private two algorithm works, you don't need that.
But you tell them there's this component which we use an algorithm that we talk about in section 3.4.
What's the output of the algorithm? What it does?
What is it on high level?
They refer to the algorithm subsection.
So they know the subsection is about the specific algorithm.
If they are not interested in how the algorithm works, they can just skip that section.
Yeah, when someone wants to implement it, they may look at the algorithm very closely.
So that's the overview.
You also need, well, the request flow might be...
Yeah, I don't know whether we need a request flow.
Given we don't have too much components, we may not need a request flow.
But we can see.
So basically there's a high level overview.
We will talk about the taxonomy and formulation.
So that's how you formulate the problem.
There are some of the content we can...
Yeah, basically everything here you can go to the formulation.
Yeah, sorry. Go ahead.
I afraid a brave diagram could have a quick view.
Do you want to add to the overview or do you want to share a screen?
Just share a screen.
Okay, I mean...
It's basically like this one.
Oh, first in the system paper, what's the classical drawing diagram software to use?
Different people use different...
Some students use GIL, some students just use Python, some use Google Sites.
Either works, whatever works the best.
We will use AI tools like Nano Banana or some other...
No, I won't suggest that. It's easy to tell.
Nano Banana works pretty well.
I use it for my class.
Some feedback I got so far is students can easily tell it's AI-generated.
Although there are a lot of details, they don't like it.
I see.
In Jack and Scroop, why did they use Nano Banana or some AI tools to generate diagrams?
Oh, you can try.
Yeah, at least the ones I generate, I feel like it's not that obvious.
It's from AI, but students can still tell.
But yeah, maybe there's a way to make it feel less like AI.
Because people certainly don't like AI-generated content.
Yeah, sure.
Yeah, give me a try.
Maybe there's some way to prompt it to make it look less like AI.
I don't know.
Go ahead.
Because we have two columns, I think we might want to change it to the horizontal one.
Where does it work?
Let's see. You can try to add it to see how it looks like.
For variables, we need minimal variables.
We need the most important variables. We certainly need them.
Because if people understand what a variable represents, what?
But we don't want too many variables.
Like, as clear as CSA, we want it.
But like, to find the equivalent, we don't want it.
Yeah, every time we show a variable, we need to tell people what it means.
We can't expect people to remember everything.
So in a paper...
Yeah, in a paper, try to minimize the use of variables and formulas.
At least, people should be able to understand the result, understanding those equations.
Okay.
So for example, our request is not necessary because we don't refer to it.
Or at least we don't have multiple requests here.
So you probably want to remove that.
All the unnecessary variables, let's remove it.
And also, font size needs to be much larger.
So when you add a figure to the paper, the font should be at least as large as the text.
So yeah, that's a bad guy and I don't have a small font.
And also, I'm not sure whether we should call it layer one.
Maybe step one.
Layer is a little bit weird.
Like, layer one, layer two.
Might be better to use step.
Or phase.
Oh, phase is clear, that's it.
So yeah, right now this goes into detail on how things work.
I think we should focus on what they do rather than how.
For example, you show primary due shadow pricing.
So you can talk about primary due and shadow pricing.
But you also need to tell people what's the goal of this component.
For example, it decides which provider it routes to.
That's the goal, right?
Yeah.
So the diagram, if you think about what information it needs, right?
So it needs input.
It gives output for each component.
So the input output is important.
How it works internally will be explained in different sections.
The diagram is to give the high level idea of how things work.
Does that make sense?
Yeah, it makes sense.
Yeah, otherwise I think that the diagram itself looks pretty nice.
Yeah.
I mean, you can also consider adding, if there's space, like adding, for example,
choose logo or some like logos of those providers.
Yeah.
Also, since we mentioned diagram, I have some questions about diagrams and system paper.
Like why read other papers?
You mentioned two diagrams yesterday.
One is system diagram.
Another is algorithm diagram.
And now they're pretty sure about the difference between these two.
Like, I'm not sure if you read this paper previously, like in this paper.
Yeah, but this is a system diagram.
This is more of details.
I will say this is more like an algorithm diagram.
Oh, it's an algorithm diagram.
So algorithm diagram basically shows the detail of how things work.
These people may not have a system.
But if you look at, say, if you go to...
Oh, maybe you recommend a paper using...
Maybe just go to an STI to pick a paper.
Oh, an STI.
Just go to the website.
Oh, website, okay.
The website doesn't have systems.
Okay.
Wait a second.
I'm not sure whether the papers are about...
Oh, it's actually available.
Do you have familiar ones?
Yeah, maybe hydrocell.
Oh, hydrocell.
I haven't read it.
Just looking at...
Oh, it's not available.
Maybe you could do an STI 25.
Okay.
I think you can schedule or participate.
Yeah, schedule, technical session.
Not opposed.
Just go through it.
Maybe just the first one.
Like predict.
This is not...
This is more of a system diagram,
but it's not very well drawn.
Maybe OCA, like the OSTI...
20...
2021?
Wait, 2021?
Or, say, yeah, OCA.
Oh, six now.
Next one.
Sorry.
Figure two?
Yes, figure two is system diagram.
It's pretty simple.
It's so simple.
I don't think it's their algorithm.
It's a figure four is their algorithm.
Okay.
Yeah.
Yeah, the diagram itself does not need to be very complex.
So try to be as simple as possible.
This is a good diagram.
This is my example.
Okay.
Well, I wouldn't say this is a good diagram.
Let me try to find a good one.
Let me see.
Yeah, do you have any paper in your mind?
I feel like I read more machine learning papers.
And I think some...
Let me think.
Yeah, some images in mind, but I don't...
I can't remember, like, which paper are they from.
Let me think about it.
Oh.
Check, like, can go...
Search can go...
SOSP.
Okay.
There's also a checkpoint paper, I remember.
What's the full name?
And groups, K-A-N-G-A-R-O-O.
Yeah, the first.
Oh, it's your paper.
Yeah, this is a single diagram.
There's also a more complex later.
Yeah, here.
This shows how requests flow
and how different components work.
It doesn't really show the details of how things work,
but to give us a high level of flow.
What are the components, how requests flow?
I see.
Okay.
Yeah, I mean, we don't need very complex ones,
like simple ones, like the one you just saw in Alka.
That also works.
Yeah, the goal is to...
When people look at the paper,
they can understand to some degree how it works.
And when people later can read the paper,
they can just look at the diagram
and tell them to remember what the paper is about.
Because there are so many papers,
no one is going to remember every detail.
So the diagram is very important.
Similarly, there's also a theorem in this paper.
But try to minimize the theorem.
I think they moved the appendix to approve the appendix.
I think, yeah, we can take a look at this paper.
You don't have to look and understand every detail.
But I think the writing of this one is pretty good.
And it was the best paper.
So this is a good algorithm diagram.
So in addition to the Markov model,
it's not an algorithm.
Here it's more for mass calculation.
For the algorithm,
yeah, we need a diagram for the algorithm.
Think about how you...
different steps of the algorithm.
Visualize the algorithm.
You can basically paste the description
into Gemini.
Ask it to generate a diagram for you.
I mean, you shouldn't use that diagram.
But it will give you an idea of what it will look like.
And polish by that initial diagram.
Yeah.
You can actually draw it behind yourself
so you can make changes.
Rather than just use the problem to make changes.
Okay.
Yeah.
Yeah.
The other thing is,
I think I don't know whether it's going to work or not.
You can give the paper to Gemini.
Ask it to generate a diagram or illustration
to see what it thinks.
It will be a fun.
I don't know whether it works.
Never tried.
But I think that the diagram you have
is already a good initial diagram.
We have built on top of it.
We already have some components
and try to iterate.
Once you finish this,
you can send us that to make it iterate.
Everyone can give you feedback on how to make it better.
Because that's the learning process.
Yeah.
I feel like the diagram is maybe something people will look at
before they read the paper.
So if there are details that appear on the diagram
that people don't understand at the first sight
or before they read the paper,
it probably shouldn't be on the diagram.
And they should understand the whole thing
after just look at the diagram.
Yeah.
Yeah.
Yes.
I think many people look at the diagram
before they read the paper.
The diagram can be,
we can have multiple diagrams.
The first one will not have detail.
The second one can have a little bit more algorithm detail.
You can think about as the second diagram,
the algorithm diagram being a ruin of the component
in the system diagram.
Now you have a primitive,
you measure a primitive in the system diagram.
Then you have an algorithm to show,
sorry, have a diagram to show how the primitive works,
whether the different inputs in the primitive
was how it works.
Yeah.
Okay.
Makes sense.
Okay.
So let's continue on the paper.
Yeah.
This all will go to the 3.2.
Yeah.
We don't want to talk about,
like in all experiments,
like something, something, something.
In design,
you only talk about the design,
like not about the experience.
Quick spot I had
when reading section two.
I feel like we give the readers a lot of assumptions,
like we treat subscription fees as some costs.
And like we also treat,
we also assume that all providers return
the same quality of work.
But I feel like we can give more justifications
on why we think of it that way.
So basically to give more kind of like persuasion
on the assumptions.
That might go into discussion.
Like for the assumptions we make,
we, so for design,
we want to make it better concise.
So we don't,
the readers don't get distracted.
Maybe I have questions about assumptions.
I think we do,
it's a good idea to provide evidence and support
on why we're making certain assumptions.
Those can usually go,
either go to background or go to discussion.
It depends on what kind of assumptions.
Yeah.
I think this is a good point.
You also, do you want to create a discussion,
like file to load down other like points
we can add to discussion?
Sure.
For example,
Yeah.
Yeah, I can do that.
Yeah.
Okay.
Yeah.
Here the amortization.
Murphy,
I think you can just imagine it in subscription
or like just as one sentence,
not as a section.
Because this is not important.
This is the assumption we make.
We need to measure it,
but we don't need a subsection.
Yeah.
This can also go to discussion.
Like modeling the local deployment as
like different type of provider.
This is actually a good discussion point.
We can move this to discussion.
No team background?
Not in background.
Because background is about things people know.
So this is your action should be sure.
Remodel it.
We think this is, we can model it as
other and a conclusive base point.
Yeah.
The background is not like you should
put your innovations or your thoughts in that.
This is certainly important,
but that's not related to design.
Maybe we have a separate section for it.
I think now let's put it in discussion.
Like later, if we have more discussion around this,
we can make a separate section.
Yeah.
The key assumptions we can make it into.
I think we need to make assumptions into the first subsection,
not the last.
Yeah.
And the optimization objective.
Yeah.
This goes to the background.
What each provider belongs to.
You can even have a table.
Yeah.
So this is the background.
I mean, everyone knows about it.
Right.
So yeah.
The optimization will be the last of this subsection.
That is the exonomy and form formulation.
Then you start to talk about the algorithm.
We probably will move the group to appendix
and just move the theorem here.
And some of the equations will also be moved to appendix.
Yeah.
Maybe we'll make this part a little bit shorter.
So offline.
Is there any limitation of pages for appendix in the SDI?
No.
No.
There's no limitation of appendix.
But people may not read it.
So your conclusion should certainly stay.
But just the details can go to appendix.
Yeah.
So those assumptions and how you award assumption.
D-rays for an hour of assumption can go to discussion.
So that's certainly not design.
So design will focus on your design.
The system and algorithm.
And also if there are assumptions.
But those are more of a discussion.
Yeah.
Then offline will be other subsection.
Online will be other subsection.
The old structure doesn't need to be changed.
But we probably need to move some of the stuff to appendix.
We don't want to scale people away with all those equations.
So one question I had regarding the learning augmented approach.
It seems to me you're just using p10 and p90.
Not really any learning, right?
Yeah.
Yeah, we use p10.
Yeah.
I think we probably should not want to call it learning augmented.
Otherwise it will backfire.
People will definitely see there's no model.
There's why it's learning.
Right?
Yeah.
Yeah.
Don't exaggerate.
That's very important.
Yeah.
We need to dial down some of the claims.
And also avoid using fancy words that's hard to understand.
So for the equations, we need to write intuition.
Like on the high level, what's the high level idea you're trying to use?
Rather than relying on people to understand how the equation works
and guess how it works on the high level.
The intuition is important.
That's what we need in writing and the diagram.
Try to use the diagram to tell the story.
Like basically try to use the diagram, like just show people the diagram
and explain using the diagram without any equation.
If you can achieve that when the diagram is ready.
Yeah.
Yeah, we don't need examples here.
I mean we don't need examples like as a subsection.
There are too many subsections right now.
We need to either remove some of them or change some of them to paragraphs.
The examples here is too now.
I think we probably will move to appendix unless it's very important.
Originally it might be appendix.
Yeah.
Some of them I moved to the main sections, but we can move it back.
Yeah.
I think just go through a movie with a few papers and go through the sections.
Try to make them understandable on a high level.
Because each time when you show an equation you need to explain every variable.
So try to minimize the equations in the main text.
Every equation you show try to explain it very clearly.
What do you try to solve?
People may not need to understand how you solve it,
but they need to understand why you have an equation,
what you are trying to achieve with the equation.
Like what we want to solve and why we want to solve.
This is much important than how we solve it.
Yeah.
How you solve it can go to appendix.
If it's super important, like if it's a new normal method,
yeah, you can have it in the main paper.
But since this is only an SDI, people probably don't care about it.
We also don't have normal mathematical methods in solving those.
So it's all existing methods.
So how you solve, you can just measure it.
You don't need to give too much detail.
Experience setup.
Are we still running any more experiments for the paper?
We certainly need more experience.
Oh, okay.
Yeah.
So if you see any experience that you think it's needed,
just send a message on Slack.
I think there are a lot of claims we don't have experience
and we need more experience.
So once you get the initial writing ready first,
then we also need to add more experience.
Because another thing I felt was like,
like from reading this without like a lot of knowledge on it,
I kind of get that the new method like works
and it is like better than the greedy or the baselines.
But I feel like we want to, this is my personal thinking.
I feel like we may also want to share like which component
of the new method is important.
Yeah.
Like right now we're showing that the entire thing as a whole
works like really well.
But I feel like we also want to share this component also
does this kind of work like a bit more specific experiments
might make things a bit better.
Yeah.
Yeah.
Yeah.
I agree.
Like we need to need a breakdown of each component.
For example, for cost and latency, we need,
first of all, we need to enter an experiment
to show how the system works
and you need more metrics like beyond just cost and latency.
You probably also want to show support
and basically all the metrics you can connect.
Then you need a breakdown into each component.
Like for example, the cost part, how you compare with offline,
how you compare with other baselines
and how many requests goes to color based,
how many goes to concurrency based
and why you reduce cost.
Those type of breakdown, as Junsu mentioned,
I think are necessary.
Then you also need to show the latency part.
Like basically giving a set of provider
how your latency modeling and hedging helps.
How many requests are hedged and how much cost it incurs.
How does it compare with simple baseline?
Like OpenRot has a few baselines.
It has a few algorithms which you need to compare with all of them.
Yeah.
Then you also need to need ablation study.
Now we will replace some traces.
What about if you have more requests
or if you have more spiky requests?
Like basically some ablation study
beyond just the trace you show.
There we need to think a little bit more carefully
what we want to present.
But yeah, I think Junsu made a good point
that we need to break down the assets
to show more understanding.
I think you had the result like a while ago
but you did not include them here, I think.
I mean at least it shows how many requests
goes to each provider.
Those type of results.
I think you had them.
Yeah.
Yeah, those sensitivity.
Yeah, you already have some.
Same for these figures.
Yeah, you shouldn't have those big figures.
You need to make them into one column.
One column.
Yeah.
So the experiment section we need.
So for example, we need to structure it
in a way that matches the design.
So basically the design you have
like basically offline optimal.
You have a section to talk about offline optimal.
That may or may not have its own subsection.
But robustness and uncertainty,
I'm not sure this is going to be a section.
It might be a subsection or maybe a paragraph.
Well, now we have too many subsections.
Subsections.
Yeah.
So the console will be a subsection.
The millions will be a subsection.
Otherwise, if you have too many subsections,
people won't understand what's the relationship
between cost and robustness.
Because we never had like super detailed analysis
on robustness.
So this probably goes to cost.
And in terms of metric,
instead of we talk about the asset,
baseline implementation.
And we also need to talk about the metrics.
So basically what metrics you use.
I'm not sure whether I want to call it comparative ratio.
You mean the cost.
But like first you mentioned system paper
does not care about comparative ratio.
But like why rate the paper like
don't be late.
That's a schedule paper.
But here I think we want to be
a little bit more clear because we have multiple metrics.
Like cost, latency, and potentially throughput.
I think showing the relative cost is more intuitive.
The relative cost is similar to comparative ratio.
Like if you show the optimal cost as one,
you can calculate.
Like instead of showing cost saving, you just show cost.
Then grading maybe like five,
or maybe 1.1 or 1.2.
Those will be easier to understand.
So your goal is to make the reader's life easy.
So they can appreciate, so they can understand
and use the idea in their own like systems.
You don't want to make it hard for readers to understand.
Otherwise they just give up.
Those I think even it needs to go to a specific
experience or section.
It shouldn't be its own section.
By looking at the structure, I don't know what it's talking about.
Is this about cost, or is it about latency,
or is it about something else?
Sensitivity can be its own section.
Yeah, I think the evaluation also needs some restructuring.
So basically you want to show an end to an experiment.
You want to show a breakdown of cost part
than the latency part.
Yeah, I think after restructuring,
we can get rid of text.
Then we can come back to intro.
Once you have everything later finished,
we can talk about the intro.
Because the intro will be decided by that.
Also, for those examples, you need a citation.
Every time you show together, you need a citation.
On the high level, I think we can talk about
more detailed writing once we have the sections in place.
Then we can talk about detailed writing.
I don't want to talk too much.
Otherwise, it's hard to remember.
Yeah, okay.
Great, yeah.
That's fair for you to drop.
I think we have finished.
Any questions?
Cool so far.
That's still a lot of things to do.
Writing is a very important skill.
Communication is super important.
We don't emphasize too much in education.
Anything like in China.
But I think presentation, both writing and speaking,
is very important.
It only has speaking requirements
and writing requirements for PhD.
Try to improve both writing and speaking.
It's also important that we interview.
Even if you're doing the job technically,
it still needs to be exhibited in some way.
Either by blogs or by talking to people.
Otherwise, people won't know about it.
That's a skill you need to practice.
I have a rough idea
by this meeting of what is a good paper.
I think if I understand it right,
I think a good paper should organize our work
into a good and logical structure
that is tailored to people's reading habits.
We usually do a breadth-first search.
We understand the higher level and narrow it down
if we are curious about it.
We use a breadth-first search manner.
Our ultimate goal is to minimize the reader's cognitive load
and that is a great paper.
A good writing needs to make reader's life easy
because your goal is to convey your ideas to others.
If it is easier for others to understand,
it is better.
Is it the same in Russian learning paper?
It is the same law.
When I read some papers in SML,
they use a lot of equations.
It depends on your innovation.
If you are designing a new method,
whether it is a mathematical method
or if your contributions are proof,
you certainly need those equations to prove it.
If it is a system or algorithm,
you want to make the system and algorithm
innovations easier for people to understand.
Although for our route-wise,
if in SML, our organization of the paper
also would be different.
It would be different.
As in SML, people like to see little bit of equations,
but still we still need the system diagrams
that will help people to understand.
You are trying to say something?
I was trying to say that clarity is definitely more important
than how many equations we have.
I think machine learning prefers more equations,
but we don't have to intentionally put
many equations in the paper.
Otherwise, it is hard to understand.
There are some equations,
but the writing should make it easy
for people to read without understanding the equation.
If everything you want to say is in the equation
and people don't understand the equation,
then they won't be able to read.
Writing takes time.
It takes iterations.
After you finish a few papers,
then you will gradually know how to write the paper.
As a reader paper,
you sometimes realize some papers are really hard to understand.
Try to see why.
Some papers are easy to understand.
You can also ask yourself why this paper is easy to understand.
Some of them, the idea is easy.
Some of them are because of writing.
The hard to understand paper is because
writing is not good enough.
If you give it to an agent,
the agent can understand it better
than writing.
Any questions?
Anything else we need to discuss?
If no, I guess we can end this meeting.
For the open letter experience,
it would be good if you could draw
against the different policies
you want to provide.
You can explicitly specify which policy you want to use.
You want to compare with all of them.
Let me know when this is finished.
Also, let me know when you finish the restructuring.
For one-week experiment,
or just one-day experiment again?
Let's do one-day so we can reach out to them earlier.
We don't have to wait for a week.
You can leave it running for one week,
but let's get one-day results so we can start to
finish the restructuring.
Let us know when you think
the paper has been restructured
and that you need help from others
to help with the writing.
Really?
I'll try to finish much earlier.
Great.
See you next week.
Bye.