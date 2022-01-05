This is my exploration into effective methods for convergence in GANs. 
# Nebula GAN
A DCGAN to generate 256 x 256 images of nebulas. 

## Details
The model was trained on approximately 800 images for 1500 epochs. Generators were saved at fixed intervals as the model suffered from minor partial mode collapse. Although the images generated at later epochs were pretty diverse, they had a few similar features in similar places.

### Things I tried to mitigate partial mode collapse
- Adding several layers of Gaussian noise to the discriminator
    - Took too long to converge due to crippling the discrminator
- Increasing the dropout probability in the discriminator
    - Took too long to converge due to crippling the discriminator
- Implemented one-sided label smoothing
    - Although there wasn't any significant decrease in the convergence rate, the model still yielded similar-looking images

## Examples from the final saved generator (1500 epochs)
Notice how the 3rd and 4th images have the same "crown" near the top-middle. Those two images exemplify the degree of partial mode collapse I was dealing with.

![Example 1](https://cdn.discordapp.com/attachments/903829974488870933/927237567219396618/test_1.jpg)
![Example 2](https://cdn.discordapp.com/attachments/903829974488870933/927237589579227277/test_2.jpg)
![Example 3](https://media.discordapp.net/attachments/903829974488870933/927237654246989864/test_98.jpg)
![Example 4](https://media.discordapp.net/attachments/903829974488870933/927237701793644544/test_75.jpg)
![Example 5](https://media.discordapp.net/attachments/903829974488870933/927237759448514680/test_9.jpg)
![Example 6](https://cdn.discordapp.com/attachments/904563989760065541/927245611143671818/test_6.jpg)
![Example 7](https://cdn.discordapp.com/attachments/904563989760065541/927245636628258877/test_11.jpg)
![Example 8](https://cdn.discordapp.com/attachments/904563989760065541/927245665782886450/test_12.jpg)