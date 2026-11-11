# pigment-aggregation-quantification

Cuantifica la **agregación / dispersión de melanóforos** (*Danio rerio*, campo claro) a partir del patrón de claroscuros de una foto, y compara imágenes de la misma región.

> Estado: prueba de concepto (v2). Falta una mayor muestra y de mayor calidad

## Contexto

Al experimentar en clase con melanóforos de Danio rerio, se me ocurrió que debería ser posible estimar numéricamente el grado de agregación y dispersión de la melanina (tras ser expuestos a diferentes compuestos) en base a imágenes de microscopio de campo claro mediante algún tipo de procesamiento, por simple que fuese (escala de grises, saturación, contraste, conteo mediante un umbral simple...), así que este proyecto es el resultado de esta idea llevada a cabo con la ayuda de LLMs.

> [!NOTE]
> Esta es la **v2** de la *pipeline*. La primera versión (rama [`v1`](https://github.com/hgex07/pigment-aggregation-quantification/tree/v1)) trabajaba con imágenes más amplias y se probó con las fotos de un experimento anterior con la misma dinámica. Como dio buenos resultados, se mantuvo, pero adaptada: al preparar las imágenes de la práctica 3 aparecieron problemas de alineamiento (sobre todo por el movimiento u orientación de la larva), así que se recortaron regiones concretas y más estables. El cambio principal es el **fondo**: la v1 usa la luminancia del portaobjetos para separar el pigmento de lo que no lo es; la v2 usa el **tejido no pigmentado de la larva** y, por eso, el **canal R**, que casi no ve el amarillo.

## Pipeline

Agregada o dispersa, la célula tiene la **misma melanina** (cambia su reparto). Agregada: poca superficie y muy oscura. Dispersa: mucha superficie y más clara. El procesamiento de cada imagen es el siguiente:

1. **Región de medida.** Se ignora la zona transparente del recorte: solo se mide donde el canal alfa es ≥ 200 (erosionado 2 px para quitar el borde difuminado). Sin esto, el exterior transparente, guardado como negro, contaría como pigmento. Todos los % son respecto a esta región.
2. **Canal R.** Se descartan G y B y se usa el canal rojo, casi "ciego" al amarillo de los xantóforos, que separa mejor el pigmento del tejido circundante.
3. **Fondo = tejido sin melanina.** Otsu separa los píxeles en oscuros (melanina) y claros; solo los claros son candidatos a fondo. Sobre ellos se ajusta un valor constante y, de forma iterativa, se descartan los candidatos claramente por debajo (residuo < −2 desviaciones típicas), sin reincorporarlos después. Resultado: un único valor de referencia por imagen.
4. **Transmitancia** = píxel ÷ referencia (1 = tejido típico sin pigmento, 0 = opaco). Compensa también las diferencias de exposición entre fotos de una misma larva.
5. **Densidad óptica** OD = −log₁₀ T, aproximadamente proporcional a la cantidad de melanina por píxel (Lambert-Beer).
6. **Máscara con dos umbrales** comunes a todas las imágenes: "semillas" con OD > umbral de Otsu global (calculado con todas las imágenes juntas) y píxeles con OD > 0,18 **conectados** a alguna semilla. Así entran las dendritas claras de los melanóforos dispersos y no las zonas oscuras aisladas del fondo; con un único umbral se perderían las regiones dispersas o entraría "basura".
7. **Métricas** sobre la máscara: área pigmentada, OD media en el pigmento, pigmento total (ΣOD), **índice de extensión** (área ÷ ΣOD), solidez, complejidad del contorno y nº/tamaño de manchas. Por la saturación de la señal y la pérdida de nitidez de algunas fotos (ver [Limitaciones](#limitaciones)), la métrica de comparación es el **área pigmentada relativa al control (Ringer) de la misma larva**, aprovechando el diseño emparejado.

Las imágenes sin canal alfa se procesan como en la v1 (fondo polinómico sobre la luminancia del portaobjetos). Todo es configurable desde la línea de comandos (`--channel`, `--bg-order`, `--low-thr`, `--seed-thr`...).

<br>

> Para una mayor robustez, se hace un barrido de 79 umbrales (`barrido_umbrales.csv`), comparación por teselas de 64 px y, en la P3, un análisis de sensibilidad con 24 combinaciones de parámetros.

<br>

## Replicación

Para obtener las imágenes se aconseja fijar tanto la cámara/ocular como la muestra para asegurar la alineación entre regiones, así como tomar las fotos en modo manual y fijar los parámetros fotográficos (ISO, exposure, WB, enfoque...). Si se recorta, usar el mismo recorte (con exterior transparente) para todas las fotos de una larva.

Para ejecutar el codigo (requiere Python ≥ 3.10):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# experimento completo de la P3 (17 recortes, 5 larvas) -> results/p3
python experiment.py data/p3 --out results/p3

# cualquier conjunto de imágenes alineadas; se comparan con la primera
python analyze.py data/p3/L3_ringer.png data/p3/L3_epinef_+10.png data/p3/L3_caf_+15.png \
    --labels ringer epinef caf --channel R --bg-order 0 --out results/L3

# las dos imágenes de prueba de la v1 (sin alfa: mismo resultado que en la v1)
python analyze.py data/img1.png data/img2.png --labels img1 img2 --out results/prueba
```

Parámetros opcionales de `analyze.py`: `--channel lum|R|G  --bg-order 2  --low-thr 0.18  --seed-thr -1  --blur-sigma 1.0  --min-object-px 15  --tile-px 64  --alpha-min 200  --erode-px 2`.

Resultados esperados de la P3 (`results/p3/metricas_experimento.csv`, área pigmentada relativa al Ringer; umbral de Otsu común = 0,557):

| Larva | Paso | Área rel. | ¿Robusto?* | Lectura |
|---|---|---|---|---|
| Control | Ringer → Epi t0 → Epi +15 → Vehículo | 1,04 → 0,96 → 0,87 | sí | agregación leve que continúa |
| L2 | Ringer → Epi 10+10 | 0,99 | no (12/24) | no agregó |
| | → Forsk +10 | 1,14 | sí | aumento, pero sin agregación previa no es rescate |
| L3 | Ringer → Epi +10 | 0,73 | sí | agregación clara |
| | → Caf +15 | 0,65 | sí | ambiguo: la forma apunta a dispersión, pero la foto está muy desenfocada |
| L4 | Ringer → Epi +10 | 0,93 | sí | agregación moderada |
| | → DBcAMP+Epi +10 → DBcAMP +10+10 | 0,77 → 0,76 | sí | sigue agregando |
| L4v2 | Ringer → Epi +10 | 1,03 | no (8/24) | no agregó |
| | → DBcAMP +10 | 1,20 | sí | aumento, pero sin agregación previa y foto borrosa |

\*Robusto = el signo del cambio se mantiene en las 24 combinaciones: canal R/lum × fondo orden 0/1 × umbral bajo 0,12/0,18/0,25 × umbral semilla global/por larva (`sensibilidad_resumen.csv`).

Con n = 1 por condición todo es descriptivo: la epinefrina agregó claramente en L3 y L4 (leve en el Control) y en ninguna de ellas se recupera el área tras el rescate. Los resultados de la v1 con las dos imágenes de prueba están en `results/prueba/` y en el [README de la rama `v1`](https://github.com/hgex07/pigment-aggregation-quantification/tree/v1#readme).

## Archivos

```
analyze.py            # punto de entrada: ejecuta la pipeline sobre N imágenes y guarda results/
experiment.py         # lote de la P3: agrupa por larva (tabla SECUENCIA), relativo al basal, sensibilidad
pigment_pipeline.py   # núcleo: región de medida, fondo, OD, máscara, métricas
figures.py            # figuras (solo dibujan)
data/p3/              # 17 recortes de la P3 (PNG con exterior transparente), alineados por larva
data/img1.png, img2.png  # imágenes de prueba de la v1 (img1 más agregada, img2 más dispersa)
results/p3/           # salida de referencia de experiment.py
results/prueba/       # salida de referencia de analyze.py con img1/img2
```

Los nombres de `data/p3` siguen `<larva>_<tratamiento>_<tiempo>.png` (p. ej. `L4_ampc+epinef_+10`: DBcAMP + epinefrina, +10 min). `results/<carpeta>/<nombre>_vertical.png` apila las etapas de una imagen en vertical, sin texto.

## Limitaciones

- n = 1 por condición.
- **Saturación:** la melanina está en 5-15/255, casi el negro de la cámara. La OD tiene techo y el pigmento total no se conserva entre fotos del mismo campo (L4: 51 → 38 al agregar). El índice de extensión equivale a 1 / OD media en la máscara y, con saturación y desenfoque, **sube al agregar** (L3, L4). Por eso aquí se usa el área relativa al Ringer y el índice se reporta como poco fiable.
- Las fotos son sRGB procesadas por el móvil (no lineales): la OD es solo aproximada.
- **El desenfoque se parece a la dispersión**: enfocar igual en todas las fotos. Las tres fotos que "dispersan" (L3_caf, L4v2_ampc, L2_forsk) son más borrosas que la anterior de su larva (columna `nitidez`).
- Alineamiento manual aproximado (correlación OD con el Ringer: 0,58-0,86; `alineamiento.csv`). Control +15 y Vehículo están desplazados ~26 px.
- L2 y L2′ se tratan como la misma larva; L4 y L4v2 como series separadas.
- El umbral bajo (0,18) se eligió a ojo con las imágenes de la v1 (revisarlo con fotos nuevas).
- La estimación de fondo necesita algo de tejido sin melanina dentro de cada recorte.
