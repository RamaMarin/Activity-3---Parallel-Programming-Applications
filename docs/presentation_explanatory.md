# Documento Explicativo para Presentacion

Este documento resume de forma explicativa que se hizo en cada ejercicio, que datos reales se usaron, como se distribuyo el trabajo paralelo y como se atendio cada task solicitada en la actividad.

## Exercise 1. Parallel Matrix Multiplication

### Dataset usado
Se uso el dataset real HB/plat362 de SuiteSparse y se complemento con bcsstk13 y west0479. plat362 es una matriz 362x362 simetrica; al expandir la simetria se trabajaron 5,786 entradas efectivas.

### Flujo de trabajo
Se compararon cuatro estrategias principales: baseline serial, particion por filas, particion por columnas y descomposicion por bloques 2D. Tambien se incluyo una variante hibrida de Strassen y un script MPI por bloques de filas.

### Cobertura de tasks
1. **Task:** Implement a serial baseline for dense matrix multiplication and validate correctness with small test cases.
   **Como se atendio:** A classical serial implementation and a NumPy serial baseline were included. Validation was done with small square matrices and error checks against NumPy.
2. **Task:** Implement parallel multiplication with Python multiprocessing using a row-based partition.
   **Como se atendio:** Rows of matrix A were split among workers. Each worker received a block of rows and the full matrix B, computed its partial product, and the final result was stacked vertically.
3. **Task:** Implement parallel multiplication with Python multiprocessing using a column-based partition.
   **Como se atendio:** Columns of matrix B were split among workers. Each worker received the full matrix A and one block of B columns, then the partial outputs were concatenated horizontally.
4. **Task:** Implement a block-based or quadrant-based parallel version and justify the design.
   **Como se atendio:** A two-dimensional block decomposition was implemented. Each task computed one C_ij block from an A row block and a B column block. The design shows how output blocks can be assembled from smaller independent products.
5. **Task:** Implement one distributed-memory version using mpi4py.
   **Como se atendio:** The file exercise_1/mpi_matrix_mult.py implements row-slab distribution with Scatterv, Bcast, and Gatherv. This covers data distribution, parallel local multiplication, and global result recovery.
6. **Task:** Implement or analyze a Strassen-based version as an advanced method.
   **Como se atendio:** A hybrid Strassen implementation was included. It pads to the next power of two and switches to NumPy multiplication below a cutoff to avoid excessive recursion overhead.
7. **Task:** Evaluate your methods on dense synthetic matrices of increasing size.
   **Como se atendio:** Dense experiments were run for sizes 64, 128, and 192 to compare the scaling behavior of the serial and multiprocessing implementations.
8. **Task:** Select at least two real sparse matrices from the SuiteSparse Matrix Collection and discuss sparsity effects.
   **Como se atendio:** Three real SuiteSparse matrices were used: plat362, bcsstk13, and west0479. Their sparsity patterns were loaded from Matrix Market archives and multiplied with sparse row dictionaries.
9. **Task:** Compare the serial and parallel versions and identify the main bottlenecks.
   **Como se atendio:** The comparison showed that Python process startup, pickling, and memory movement dominate for these matrix sizes. Sparse workloads also introduce irregular work per row, which affects balance.

### Puntos clave para exposicion
- Las pruebas densas se hicieron con tamanos 64, 128 y 192 para observar el efecto del overhead.
- Las matrices dispersas reales mostraron que la estructura de no ceros cambia el balance de carga.
- En este entorno local, NumPy serial fue mas rapido que multiprocessing para tamanos medianos y pequenos por el costo de mover datos entre procesos.

## Exercise 2. Parallel Cell Image Processing and Morphological Characterization

### Dataset usado
Se uso el dataset real DIC-C2DH-HeLa con secuencias ['01', '02'] y 168 cuadros anotados. Las imagenes son TIFF en escala de grises de tamano [512, 512].

### Flujo de trabajo
Cada worker procesa una pareja imagen-mascara. La mascara silver-truth se toma como segmentacion base; despues se extraen objetos etiquetados y se calculan bounding box, area y ejes mayor y menor.

### Cobertura de tasks
1. **Task:** Download and inspect the DIC-C2DH-HeLa dataset. Describe the image format, image size, and any relevant metadata for measurement.
   **Como se atendio:** The real dataset was extracted locally. The analyzed frames are grayscale TIFF images of size [512, 512] and measurements were reported in pixels.
2. **Task:** Build a serial pipeline that reads an image, segments cells, extracts connected components, and computes object-level measurements.
   **Como se atendio:** The serial pipeline reads one raw frame and one labeled mask, extracts every nonzero labeled region, and computes morphology for each detected cell.
3. **Task:** Use a pretrained model such as Cellpose if it improves the segmentation quality. Document the model used and the main inference settings.
   **Como se atendio:** Instead of Cellpose, the workflow used the dataset's silver-truth segmentation masks from *_ST/SEG. This provided a higher-confidence segmentation reference for morphology measurement and removed model-inference variability.
4. **Task:** For each detected object, compute bounding box, area, major axis length, and minor axis length.
   **Como se atendio:** Each labeled cell yields an axis-aligned bounding box, pixel area, and PCA-based major and minor axis lengths.
5. **Task:** If you choose to compute rotated bounding boxes, describe how they are obtained and provide visual examples.
   **Como se atendio:** Rotated boxes were not required because they were optional. The implementation focused on axis-aligned boxes, which were sufficient for the requested quantitative summary.
6. **Task:** Parallelize the pipeline with Python multiprocessing and explain the work distribution.
   **Como se atendio:** The work was distributed by image because frames are independent. Each worker receives one image-mask pair, computes measurements locally, and returns object-level and image-level summaries.
7. **Task:** Generate a table of summary results per image, including count, average width, average length, and variability.
   **Como se atendio:** CSV outputs were generated per image and per sequence. The summary tables report detected cells, average width, average length, and standard deviations.
8. **Task:** Measure and compare serial and parallel execution times for different numbers of workers.
   **Como se atendio:** The real dataset was benchmarked with 1, 2, and 4 workers over 168 annotated frames.
9. **Task:** Discuss limitations of the chosen segmentation approach and how segmentation quality affects the final measurements.
   **Como se atendio:** The discussion explains that morphology quality depends directly on mask quality. Using silver-truth masks makes the measurements more trustworthy, but it removes the cost and uncertainty of learned segmentation.

### Puntos clave para exposicion
- La distribucion por imagen fue la opcion natural porque cada frame es independiente.
- Con 168 cuadros anotados, el paralelismo si dio mejora: 1.34x con 2 workers y 1.87x con 4 workers.
- La calidad de la medicion depende directamente de la calidad de la mascara; por eso se usaron mascaras silver-truth del propio dataset.

## Exercise 3. Forest Fire Cellular Automaton Driven by NASA FIRMS Data

### Dataset usado
Se uso el archivo real modis_2024_Mexico.csv con 80390 hotspots MODIS para Mexico. Para el benchmark se filtraron 31473 registros entre 2024-03-01 y 2024-05-31.

### Flujo de trabajo
El flujo fue: filtrar detecciones, convertirlas a una grilla regular, construir un calendario temporal de igniciones externas y ejecutar el automata celular con vecindad de Moore. La version paralela divide la grilla en franjas horizontales.

### Cobertura de tasks
1. **Task:** Obtain hotspot data from NASA FIRMS for a selected region and time window. Document the source, filtering criteria, and variables used.
   **Como se atendio:** The file exc3_modis_2024_Mexico.csv was used as a Mexico MODIS yearly summary. The benchmark filtered the data to 2024-03-01 through 2024-05-31 with confidence >= 70. Variables used included latitude, longitude, FRP, confidence, and acquisition date.
2. **Task:** Transform the hotspot detections into a regular two-dimensional grid suitable for a cellular automaton.
   **Como se atendio:** The selected detections were normalized to the Mexico bounding box of the filtered data and mapped into regular 240x240 and 420x420 grids.
3. **Task:** Define the state space, neighborhood structure, ignition rule, and transition logic.
   **Como se atendio:** States 0, 1, 2, and 3 represent non-burnable, susceptible vegetation, burning, and burned. A Moore 8-neighborhood was used. Burning neighbors and FRP-based intensity increased ignition probability, and external MODIS detections injected new ignitions over time.
4. **Task:** Implement a serial simulation and verify that the model evolves coherently over time.
   **Como se atendio:** A serial version was implemented and verified through consistent burned-cell counts and visual snapshots across time.
5. **Task:** Implement a parallel version with mpi4py using domain decomposition and border exchange.
   **Como se atendio:** The file exercise_3/mpi_fire_ca.py implements row-slab domain decomposition. Each process exchanges top and bottom halo rows with neighbors before updating its local subdomain.
6. **Task:** Run simulations for different grid sizes or time horizons and compare runtimes.
   **Como se atendio:** Two configurations were benchmarked: 240x240 with 60 steps and 420x420 with 90 steps, comparing serial execution against multiprocessing domain decomposition.
7. **Task:** Visualize the temporal evolution of the fire.
   **Como se atendio:** The simulation exports selected fire-state snapshots to exercise_3/results/snapshots, showing ignition, spread, and burned territory over time.
8. **Task:** Discuss the interpretation of NASA FIRMS data and the difference between hotspots and the true fire perimeter.
   **Como se atendio:** The document explains that hotspots are thermal detections at satellite overpass time, not exact perimeters. They should be interpreted as ignition evidence or activity indicators, not full fire outlines.
9. **Task:** Reflect on the scientific limitations of the simplified automaton and suggest improvements.
   **Como se atendio:** The discussion covers missing effects such as wind, slope, vegetation classes, fuel moisture, suppression, and geospatial land-cover constraints.

### Puntos clave para exposicion
- Se usaron estados discretos 0, 1, 2 y 3 para no combustible, vegetacion susceptible, fuego activo y celda quemada.
- Las capturas exportadas permiten explicar visualmente como avanza la propagacion.
- El resultado principal aqui es metodologico: la version paralela local no supero al serial porque el update serial vectorizado ya era muy rapido y el overhead de coordinacion peso mas.

## Exercise 4. Parallel K-Means Clustering

### Dataset usado
Se uso el dataset real Covertype. El archivo completo contiene 581012 filas y 54 variables predictoras; para benchmarking reproducible se trabajo con un subconjunto deterministico de 150000 muestras.

### Flujo de trabajo
Primero se estandarizaron las caracteristicas. El baseline serial ejecuta asignacion y actualizacion de centroides. La version paralela local comparte el dataset en memoria compartida y reparte rangos de filas entre workers; la version MPI reparte bloques de observaciones y sincroniza centroides con Allreduce.

### Cobertura de tasks
1. **Task:** Load the Covertype dataset and describe its size, number of features, and preprocessing.
   **Como se atendio:** The real Covertype dataset was extracted from exc4_covertype.zip. The full dataset has 581012 rows and 54 predictor features. A deterministic subset of 150000 samples was used for repeated benchmarking, and features were standardized.
2. **Task:** Implement a serial K-means baseline and validate assignments and centroid updates.
   **Como se atendio:** A serial K-means baseline was implemented with assignment and centroid-update steps. Correctness was checked by stable inertia reduction and consistent centroid updates across iterations.
3. **Task:** Implement a parallel K-means version with mpi4py and explain data distribution and centroid synchronization.
   **Como se atendio:** The file exercise_4/mpi_kmeans.py distributes row blocks of the dataset across MPI ranks and synchronizes centroids after each iteration. A local multiprocessing version was also benchmarked in this environment.
4. **Task:** Use collective communication operations to aggregate local cluster statistics and update centroids.
   **Como se atendio:** The MPI design uses Allreduce to combine local centroid sums and cluster counts. The multiprocessing version returns local sums, counts, and inertia from each worker for host-side aggregation.
5. **Task:** Test the algorithm with different numbers of clusters and different numbers of processes.
   **Como se atendio:** The benchmark used k=4 and k=7 and compared 1, 2, and 4 workers in the local shared-memory version.
6. **Task:** Measure runtime per iteration, total runtime, convergence behavior, and changes in quality or stability.
   **Como se atendio:** The outputs record iteration counts, runtime per iteration, total runtime, inertia, and convergence behavior for every tested configuration.
7. **Task:** Compare the serial and parallel versions and discuss when the parallel approach becomes advantageous.
   **Como se atendio:** For the tested local subset, the multiprocessing version stayed slower than the serial version because synchronization and worker overhead remained important. The document explains that larger datasets and truly distributed memory can shift that balance.
8. **Task:** Identify the main communication costs and discuss how dataset size and number of clusters influence scalability.
   **Como se atendio:** The main communication cost is the repeated aggregation of kxd sums and k counts each iteration. Larger datasets favor parallel execution, while larger k increases both arithmetic and synchronization volume.

### Puntos clave para exposicion
- Se probaron dos valores de k: 4 y 7.
- La distribucion de workers fue 1, 2 y 4 procesos en la version local compartida.
- El documento explica que K-means se beneficia mas cuando N es grande y la comunicacion queda relativamente pequena frente al costo de calcular distancias.
