# Comments on report (89/100)

## Abstract

Nice abstract!

## Intro (-5 marks)

When you refer to "this paper" it seems that it's the deep expander network
paper. While this can be implied, you should still put a reference to be
explicit.

In the intro, it should also talk briefly about your findings summar, as you
did in the abstract. Introduction alone should give you the big picture on what
you are expecting to see from this paper.

## Contributions of the original paper

Good and succint summary of what each paper's contributions are about. I
appreciate you also went to summarize the lineage of these architectural
developments.

## Contributions of this project (-5 marks)

It is interesting that your number of parameters were set based on actual
computation time. Was this done in the original paper as well? This could've
been made more clear.

## Results & Conclusion (-1 mark)

I would assume that it you were using the testset, which I confirmed through
looking at the code, but this could've been made explicit.

I think the results are a bit inconclusive because it is not clear whether
training has converged. As data augmentation was added, I think you could've
allowed for more training epochs. The results graph also suggest that you have
yet to converge.

Typically, when comparing betwee different architectures, performance after
convergence should be considered.


# Comments on code (93/100)

Coding could've followed [PEP8](https://www.python.org/dev/peps/pep-0008/) for
better readibilty.

Having duplicate entries in code is not good. For example, you could've simply
loaded the networks folder as a module and `getattr` on this module to avoid
having to do if-else for the architecture.

It is also a good idea to use `os.path.join` when you modify paths. This
prevents you from possible mistakes due to the lack of `/` in your path.

I think mostly everything in the paper seems to be reproduced, with the
exception of experiments being run properly. 2 marks are deducted for this
reason.


## Grading guidlines

Required a full implementation from scratch, as well as being a project that
requires heavy effort (multiple components and formulations)

- 100 for full reproduction + going beyond
- 95 for full reproduction
- 90 for partial reproduction
- 80 for implementation trains, gives results
- 70 for implementation runs


## Presentation (each item is maximum 5)

Understanding, Presentation, Slides

The second slide is too dense with text which make it hard to
follow. Preliminary works could’ve been first visualized then
summarized. Slides can be more visual rather than being text-based

U(4) P(3) S(1)
