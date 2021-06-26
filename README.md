# Simulating STM data

This Python library was created to help with the simulation of scanning tunneling microscopy data. Files can be exported as .png image file or as Nanonis .sxm files. This library provides various methods for Image creation, regarding particle arrangements or image error simulations.

This library was developed by J. Tim Seifert the course of a Bachelors thesis at IAP, TU Braunschweig.

Mail: johannes.seifert@tu-braunschweig.de

# Required Packages

 - numpy 
 - scipy
 - pickle
 - pillow, PIL
 - tqdm (If progressbar should be displayed)
 - matplotlib (Used in some testing methods)

# Performing a simulation

Generating an STM Image can be very simple:
```python
fn_gen = FilenameGenerator()
dat = DataFrame(fn_gen)
dat.add_Ordered()
dat.get_Image()
dat.save()
```
What happens in detail:
At first, a `FilenameGenerator` Instance is created. This instance helps with creating unique filenames for saving data, especially when multiprocessing is used. 
```python
fn_gen = FilenameGenerator()
```
Secondly, a new `DataFrame` instance is created.  The `DataFrame` is the basis of this simulation. It abstracts the sample and holds the particles absorbed onto it.
A bare sample is  (in most cases) not very interesting to display. Instead, molecules are bought onto the surface. Details are discussed in the chaper "Adding Particles".
Particles can be added in an ordered way using
```python
dat.add_Ordered()
```
The calculations needed to create the actual image are only performed when 
```python
dat.get_Image()
```
is called. If this call is missing, saving the image would only show a black sample. Saving is simply done by calling
```python
dat.save()
```
> Note: If no *settings.ini* is in the current directory, the program will create one and abort execution. Rerun to perform the simulation.

# Settings
Many simulation aspects are customizable. For this purpose, the *settings.ini* file is used. If the program is run without a settings file in the current context, a default settings file will be created and the program exits, otherwise the settings are read and fed into the simulation. 
The most imprtant settings are explained in the follwing table:
> Note: When boolean values are used as parameters, 0 indicates `False`, 1 (and other nonzero integers) indicates `True`.

|Setting| Default | Description  
|--|--|--|
| threads | *1  | Thr andeads used for multithreaded execution |
| ...Folder | *2 | Path to folder where Image/SXM/Data files will be stored |
| Pixel per Angström | 2 | Image resolution |
| Image width | 100 | Determines width of created image in Angström |
| Suffix/Prefix... | *3 | Determines prefix and suffix for created files |
| color scheme | WSXM | Switches between grayscale and WSXM color scheme |
| Pixel overlap | 40 | No of pixels particles can align outside the image |
| Particle width | 0.4 | Determines default particle dimensions in Angström |
| Molecule Style | simple | Switches between simple (rectangle) and complex(atomic) molecule visualization |
| Maximum height | 7.3 | Maximum height (in Angström) which is mapped to maximum brightness (255 grayscale) |
| Exponent 1/kbT | 0.4 | Determines how sharp figures are displayed. High values lead to sharp edges |
| Variation in pos. | 0.05 | Maximum relative position (in percent) parts are shifted from original position (when variation in ordered position is True) |
| Stretching factor image shift | 1.05 | Factor the image length/width is stetched to (when using image shift) |
| Shift style | Exp | Switches between linear (Lin) and exponential (Exp) image shift.|

*1: Uses CPU count of current system
*2: Uses path to program and subfolder *bildordner*, *sxm* and *data*
*3: Used namescheme: *ImageX.png*, *ImageX.sxm*, *DataX.txt*

# Particles

There are different classes to abstract particles brought onto the surface:

## Particle

The Particle class is the base class of all  particles. They are visualized as rectangles with smeared out borders using Fermi-Dirac density distribution. The particles dimensions are defined in the settings file.
How much smeared out the particles are is also defined by the Fermi-Exponent 1/kbT defined in the settings file.
If an image file for particle visualization is specified in the settings file, the particle will look like the imported image. Note that the method does not scale the imported image. The file dimensions need to fit the particle width and length.

## Molecule

Molecules can be used as Particles too. They are combined out of Atoms, modeled as instances of the `Atom` class. 
Many different types of molecules can be used. Some types are already defined. Which one is selected is controlled by the `molecule_class` parameter.

 - "Single": A single atom
 - "CO2": Adds atoms to from a carbon dioxide molecule
 - "Star": Adds atoms in a star shape: One in the middle, and four more atoms in a right angle.
 - "NCPhCN": Most important class. Modelles dicarbonitril-polyphenyl molecules. The number of phenyl groups is set by the `molecule_ph_groups` parameter.

Molecules can be displayed in two modes: "single" abstracts the molecule as a  rectangle, width and length are calculated from the maximum atomic distances. Mode "complex" shows each single atom of this molecule as a cirlce. 

## Defining own particles
Own particles can be created by extending the `Particle` class. For correct behavior, the length and width parameters have to be overridden.
Most important is the `visualize_pixel(x, y)` method. It controls, how the particle will look like. It should return the grayscale height for this particle (0 to 255) at position (x, y). The center of the molecule has coordinates (0, 0). For example: A circular particle with radius 5px without smeared out borders can be displayed by using `return 255 if ` $\sqrt{x^2 + y^2} \leq 5$ `else 0`. 


# Arrangement

The `DataFrame` provides various methods to add particles. 
For measuring positions, distances and everything that has the dimension of a length, the class `Distance` is used, which stores the length in the units Pixel and Αngström simultaneoulsy.  The constructor takes a boolean value indicated if the length is defined in Angström (`True`) or in pixel (`False)`, and the length in the specified unit. It implements magic methods to allow ordering and caluclations with it. The conversion factor between pixel and Angström is specified in the settings. Example: 
```python
# px_per_angstrom = 2
d1 = Distance(True, 2) # Distance of 2 Ang.
print(d1.px) 
-> 4
print(d1.ang)
-> 2
d2 = Distance(False, 6) # Distance of 6 pixel (=3 Ang)
d3 = d1 + d2
print(d3)
-> "Distance(10px, 5Ang)
print(d3 >= d1)
-> True
print(d1/d3)
-> 0.4
```
> Note: in the following, multiple function will be introduced. In some cases, not all possible parameters are mentioned here. The parameters not used are often irrelevant or should not be used anymore.
> 
## Manually add particles
Particles, Molecules and other displayable Objects can be added to the `Dataframe` with the `addParticle(part=None)` method. By creating a particle,
```python
x = Distance(True, 20)
y = Distance(True, 30)
theta = np.pi/2
p = Particle(x, y, theta)

dat = DataFrame(fn_gen) # see above
dat.addParticle(p)
dat.addParticle()
```
The first usage of `addParticle` adds the previously specified Particle 20 Ang. right and 30 Ang. below the top left corner of the frame with turned by an angle of 180 degree.
The second usage has no provided Particle and therefore adds a Particle at a random position inside the frame.
Molecules can be added in the same way:
```python
x = Distance(True, 20)
y = Distance(True, 30)
pos = np.array([x, y])
theta = np.pi/2
molecule_class = "NCPhCN"
molecule_ph_groups = 3
m = Molecule(pos=pos, theta=theta, molecule_class=molecule_class, molecule_ph_groups=molecule_ph_groups)

dat = DataFrame(fn_gen)
dat.addObject(m)
```
This adds a $\text{NCPh}_3\text{CN}$ molecule to the frame at specified angle and position. Due to the variety of parameters, there is no default `addObject()` method and the parameters passed to `Molecule` are keyword based.

## Adding particles randomly
Particles can also be added at random positions.  `DataFrame` provides the method `addParticles(amount=None, coverage=None, overlapping=False)`.  
If `amount` is specified, the provided number of Particles are randomly distributed added to the frame. Otherwise, if `coverage`is specified, Particles are added until the provided percentage of the frame is covered with Particles. If none of both is specified, the number specified in the settings folder is added.
The `overlapping` parameter controls, if the added particles can overlap. If `False`, before adding a particle, it is checked whether this would overlap with any existing one. If so, it would be discarded and another would be created.
## Adding ordered molecules
The `add_Ordered(theta=None, factor=1.0, ph_grps=None)`
method adds $\text{NCPh}_x\text{CN}$ molecules in a way that is observed in experimental data. For the number of phenyl groups, numbers 3 to 5 are supported. 
The `theta` parameter cotrols the orientation of the structure. If it is `None`, the orientation will be random.
The parameter `factor` allows to stretch the spacing between molecules. For physically correct ordering, it should be kept at 1.
The`ph_grps` parameter controls which molecules are used for the ordered structure. If `None`, a random number of 3, 4 and 5 is used.

# Imaging errors
To make the created images more realistic, various imaging errors are implemented.
Which errors are used and tp which extend is controlled via the *settings.ini* file entriely.
## Image Noise
Different Noise levels can be applied to the Image. White Noise and 1/f noise both can be enabled in the settings. The parameter *"Grayscale value of mean noise"* defines thea average brightness of the image for white noise and 1/f noise. The parameter "*image noise standard derivation*" defines the standard derivation used for the noise. The higher this value, the higher the difference between brightest and darkest spot of noise.
## Scanning Lines
The parameter *"Use scanlines"* adds scanlines typically seen in STM data. The implementation is quite simple. Has a much smaller effect than using 1/f noise
## Particle Dragging
The effects of a STM tip colliding with adsorbed particles and dragging them across the sample is implemented, too. The possibility of a particle to be dragged is set by the *"dragging possibility"* setting. Direction and average distance can be set by *"Raster angle"* and *"Dragging Speed"*.
> Note: It has to be assured that dragged particles still do not overlap with other particles. This leads to problems in a densely packed structure, e.g. when add_Ordered is used. It is possible that a particle cannot be dragged.

## Double Tips

An STM with two tips is implemented aswell. The probability of this to happen is set by the *"possibility of two tips"* setting. 
For this purpose, The classes `DoubleFrame` and `DoubleParticle`are implemented. Their usage is however automatic done by the `DataFrame`.

## Atomic Step

Atomic steps in the underlying metal occur with the possibility *"possibility of atomic step"*. The height can also be set in the settings file.
Note that the parameter for maximum height has to be set properly too to avoid unwanted behavior.

## Dust

Dust particles -implemented in the so called class - can be added to the frame too. The average number of dust particles is set by *"Medium no of dust particles"*.
## Image Shift
Finally, an internal stretching of the image due to nonlinear piezo behavior is implemented as image shift. The stretching factors in x- and y- direction can be adjusted seperately. Furthermore, the Style of the shift can be switched between exponential and linear mode.
