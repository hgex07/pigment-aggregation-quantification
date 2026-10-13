# pigment-aggregation-cuantification

Cuantifica la **agregación / dispersión de melanóforos** (*Danio rerio*, campo claro) a partir del patrón de claroscuros de una foto, y compara imágenes de la misma región.

> Estado: prueba de concepto. Falta una mayor muestra y de mayor calidad

## Contexto

Al experimentar en clase con melanóforos de Danio rerio, se me ocurrió que debería ser posible estimar numéricamente el grado de agregación y dispersión de la melanina (tras ser expuestos a diferentes compuestos) en base a imágenes de microscopio de campo claro mediante algún tipo de procesamiento, por simple que fuese (escala de grises, saturación, contraste, conteo mediante un umbral simple...), así que este proyecto es el resultado de esta idea llevada a cabo con la ayuda de LLMs.

## Pipeline

Agregada o dispersa, la célula tiene la **misma melanina** (cambia su reparto). Agregada: poca superficie y muy oscura. Dispersa: mucha superficie y más clara. Al medir las dos cosas simultáneamente y hacer el cociente se puede obtener la métrica principal, el **índice de extensión** (área pigmentada ÷ pigmento total**). El procesamiento de las imágenes se realiza en diferentes fases:

| # | Paso | Para qué |
|---|------|----------|
| 1 | Escala de grises (luminancia) | un número por píxel |
| 2 | Estimar el fondo (polinomio de grado 2 sobre los píxeles claros, 4 iteraciones) | modelar la iluminación sin células |
| 3 | Transmitancia = gris ÷ fondo | corregir la iluminación irregular |
| 4 | Densidad óptica OD = −log₁₀(transmitancia) | valor proporcional a la cantidad de melanina (Beer-Lambert) |
| 5 | Máscara[^1] por umbral doble | decidir qué es pigmento |
| 6 | Métricas[^2] | resumir cada imagen en números |

[^1]: Un píxel es pigmento si OD > 0,18 **y** está conectado a un píxel con OD > umbral de Otsu (calculado con todas las imágenes juntas). Así entran las dendritas claras y no las sombras sueltas del fondo. Ambos umbrales son comunes a todas las imágenes.

[^2]: área pigmentada, OD media, pigmento total, **índice de extensión**, solidez y complejidad del contorno, nº y tamaño de manchas (estos dos, poco fiables con células que se tocan).

> Para una mayor robustez, se hace un barrido de 79 umbrales (`barrido_umbrales.csv`) y comparación por teselas de 64 px.

## Replicación

Para obtener las imágenes se aconseja fijar tanto la cámara/ocular como la muestra para asegurar la alineación entre regiones, así como tomar las fotos en modo manual y fijar los parámetros fotográficos (ISO, exposure, WB, enfoque...).

Para ejecutar el codigo (requiere Python ≥ 3.10):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python analyze.py data/img1.png data/img2.png --labels img1 img2 --out results
```

Parámetros opcionales: `--low-thr 0.18  --bg-order 2  --blur-sigma 1.0  --min-object-px 15  --tile-px 64`.

Resultados esperados (img2 ÷ img1) en `results/metricas.csv`:

| Métrica | img1 | img2 | img2 ÷ img1 |
|---|---|---|---|
| Área pigmentada | 22,8 % | 38,3 % | 1,68 |
| OD media en pigmento | 0,52 | 0,42 | 0,82 |
| Pigmento total (kOD) | 17,2 | 23,5 | 1,37 |
| **Índice de extensión** | **1,93** | **2,37** | **1,23** |
| Solidez media | 0,86 | 0,79 | 0,91 |

Umbral de Otsu = 0,274. Las curvas del barrido se cruzan en OD ≈ 0,49: por debajo img2 tiene más área (pigmento extendido), por encima img1 (pigmento concentrado). Teselas con más pigmento en img2: 22 de 30.

## Archivos

```
analyze.py            # punto de entrada: ejecuta todo y guarda results/
pigment_pipeline.py   # núcleo: fondo, OD, máscara, métricas
figures.py            # figuras (solo dibujan)
data/                 # img1 (más agregada), img2 (más dispersa): recortes alineados a mano
results/              # salida de referencia (tablas y figuras)
```

`results/<nombre>_vertical.png` apila las etapas de una imagen en vertical, sin texto.

## Limitaciones

- n = 1 por condición.
- El pigmento total difiere un 37 % entre las dos imágenes: los recortes no contienen exactamente lo mismo, así que el área bruta (×1,68) exagera; el índice de extensión (×1,23) es la cifra a usar.
- Alineamiento manual aproximado (correlación OD píxel a píxel: 0,54).
- **El desenfoque se parece a la dispersión**: enfocar igual en todas las fotos.
- El umbral bajo (0,18) se eligió a ojo con estas dos imágenes (revisarlo con fotos nuevas).
- La estimación de fondo necesita ver algo de tejido sin pigmento en cada recorte.
