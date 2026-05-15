# Five Minutes Ahead

Last June, junior Jack Gao was attending a science camp in Oklahoma when five tornadoes ripped through the area in a single week. He came home interested in the warnings.

"How we issue tornado warnings is god-awful," Gao said. "Every day the National Weather Service will be like, 'Here's a general risky area,' and this area will be the size of a state. From there, it's just people on the ground spotting the tornado."

A tornado watch arrives five or six hours before severe weather. The actual warning, he explained, lands later.

"We're able to give people five to six hours of lead time on a watch, but the average lead time of an actual tornado with the tornado warning is ten to fifteen minutes," Gao said. "The primary reason we still even have ten to fifteen minutes is because, on average, the tornado is ten to fifteen minutes."

He put it plainly.

"We're like, it's five miles away, it takes ten minutes to get to us, and that's your lead time," Gao said. "We don't actually predict anything."

Gao has worked at George Mason University's Center for Ocean, Land, and Atmosphere Studies since freshman year, first on ocean heat flux, then on a spatial-temperature analysis whose figures landed in Virginia's official climate assessment. After Oklahoma, he turned to tornadoes. He saw the scale problem first.

"Tornadoes are really small in terms of atmospheric structures. They average 200 meters across," Gao said. "The highest resolution generative forecast we have is five by five kilometers."

Standard models evolve the atmosphere forward in time. Tornadoes are too small.

"Even someone clapping their hands creates little eddies in the air, and that will propagate and affect stuff at this scale," Gao said. "You might be somewhat accurate for the first fifteen seconds, but by the time you get to ten to fifteen minutes, it's a butterfly effect."

Gao threw out the prediction-by-simulation playbook.

"This is the first ever model that combines convolutional neural networks with random forest," Gao said. "Random forest models are good because they're very good at making decisions. Convolutional neural networks, how I thought of them, were less of making decisions of yes/no and more of feature extraction."

A principal-component step bridges the two, feeding spatial features from the CNN into the decision tree. Researchers at MIT's Lincoln Laboratory built the tagged Doppler radar dataset he trains on; Gao bolted on a polar-coordinate range mask because radar doesn't think in Cartesian grids. The National Weather Service had tried something similar in 2019.

"A CNN was made by the National Weather Service in 2019," Gao said. "But it completely ended up being worse than the current model. They just did a straight CNN."

Gao posts a 68 percent improvement on CSI, the gold-standard metric for tornado forecasting. He ran the model live against the 2023 Memphis outbreak, the third-deadliest in U.S. history.

"I successfully predicted 13 of the 14 tornadoes," Gao said. "The maximum prediction time was 20 minutes ahead of a storm forming. It averaged around seven minutes."

He worried the model was cheating: maybe it caught weak tornadoes that already existed instead of predicting new ones. He checked the timing of its hits.

"I saw the vast majority of my positive hits before there were any actual tornadoes," Gao said. "When the actual tornadoes were maturing, I saw a lot lower positive rate. That's big."

Working with his professors and a National Weather Service coach, Gao thinks his ablation analysis is the first observational evidence of the Rotunno-Klemp-Wiseman hypothesis, a forty-year-old theory of how horizontal vorticity tips vertical to form a tornado.

"It was developed in the 1980s, and it's never been observed in real life in tornadoes before," Gao said. "Because who's going up to a tornado?"

He has moved off his laptop onto a supercomputer; his team has asked him to point the same approach at dust storms.

"Tornadoes are sort of like the million-dollar problem of weather forecasting," Gao said. "If you're able to crack tornado forecasting, you can apply that model to so many other things, because it's solving the inherent paradox of high resolution and low error rate. Those are fundamentally opposing forces. But if you're able to develop a model that's able to do this, then that's just going to be the best model for everything."
