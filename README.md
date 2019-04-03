# CelebAMask-HQ
![image](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/images/front.jpeg)
We build a large-scale face semantic label dataset named CelebAMask-HQ, which was labeled according to CelebA-HQ that contains 30000 high-resolution face images.
CelebAMask-HQ was precisely hand-annotated with the size of 512 x 512 and 19 classes including all facial components and accessories such as skin, nose, eyes, eyebrows, ears, mouth, lip, hair, hat, eyeglass, earring, necklace, neck, and cloth. 
For occlusion handling, if the facial component was partly occluded, we label the residual parts of the components.
On the other hand, we skip the annotations for those components that are totally occluded.
