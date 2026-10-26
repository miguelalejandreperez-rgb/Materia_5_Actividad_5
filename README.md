[Actividad5_README.md](https://github.com/user-attachments/files/33952056/Actividad5_README.md)
# Actividad 5 | Bank Marketing (Kaggle) + MLflow

**Materia:** Gestión de Proyectos de Inteligencia Artificial — Universidad Tecmilenio  
**Autor:** Miguel Ángel Alejandre Pérez  
**Dataset:** [Banking Dataset – Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets) (`prakharrathi25/banking-dataset-marketing-targets`).

## 1. Objetivo del proyecto
Comparar regresión logística contra Random Forest para pronosticar la suscripción a un depósito bancario (`y=1`, sí; `y=0`, no). El foco de selección es **PR-AUC en validación cruzada** por el desbalance de clases; accuracy, precision, recall, F1 y ROC-AUC se presentan como métricas complementarias. El resultado no constituye una solución comercial lista para desplegar.

## 2. Resultados **observados** (Colab, 26-10-2026)

| Indicador (prueba, n=9,043) | Regresión logística | Random Forest |
|---|---:|---:|
| Accuracy | 0.8933 | **0.8948** |
| Precision positiva | 0.6742 | **0.6931** |
| Recall positivo | 0.1701 | **0.1815** |
| F1 positivo | 0.2717 | **0.2876** |
| ROC-AUC | 0.7519 | **0.7908** |
| PR-AUC (average precision) | 0.3960 | **0.4433** |
| PR-AUC (CV, 5 pliegues) | 0.3866 | **0.4316** |

En el test, Random Forest presenta 192 TP, 85 FP, 7,900 TN y **866 FN**. McNemar exacto: p=0.1843. No se concluye superioridad estadística en tasa de aciertos al 5%; el modelo preferido se elige por PR-AUC en CV, no por explorar el test.

## 3. Estructura

```text
Actividad5/
├── datos/
│   ├── datos_ini/train.csv
│   └── datos_limp/bank_marketing_limpio*.csv
├── fuentes/
│   ├── datos_prep.py
│   ├── train.py
│   └── verificar_evidencias.py
├── entrena.ipynb
├── manifest_dataset.json
├── versiones_entorno.json
├── resultados/{tablas,graficas}/
├── evidencias/
├── mlruns/
├── Reporte_Actividad_5.docx
├── Reporte_Actividad_5.pdf
├── README.md
├── CHANGELOG.md
└── requirements.txt
```

## 4. Ejecución en Google Colab (recomendado)

1. Abre [Google Colab](https://colab.research.google.com/) y sube `entrena.ipynb`.
2. Selecciona **Entorno de ejecución → Ejecutar todas**. No es necesaria GPU.
3. La celda inicial instala `kagglehub` y `mlflow`; las siguientes descargan `train.csv` y crean `/content/Actividad5/`.
4. Verifica al final los dos modelos, las métricas y los directorios `resultados/`, `mlruns/`, `evidencias/`.
5. Para evidencias del **panel auténtico**, ejecuta la celda `11. Acceso opcional al panel visual de MLflow` y captura lista de corridas, parámetros, métricas y artefactos. No se proporcionaron capturas del panel en el ZIP recibido, pero sí archivos persistidos de MLflow.
6. Descarga el ZIP generado por la última celda; la presente entrega ya incorpora los resultados de esa ejecución.

### Ejecutar desde consola (opcional)

```bash
cd Actividad5
python -m pip install -r requirements.txt
python fuentes/verificar_evidencias.py
python fuentes/datos_prep.py
python fuentes/train.py
```

`train.py` reproduce el procedimiento básico de entrenamiento, pero crea **corridas nuevas** en el `mlruns` local; no debe usarse para sustituir resultados sin registrar su procedencia. `datos_prep.py --descargar` vuelve a obtener el CSV desde Kaggle con `kagglehub` si se dispone de red.

## 5. Limpieza, reglas y versionado

- Normalización de nombres y categorías; `unknown` en campos categóricos → valores ausentes.
- Los ausentes **no se imputan globalmente antes de dividir**. La imputación está dentro de `Pipeline` y se aprende en cada pliegue para evitar fuga.
- Se comprobaron duplicados exactos (0), etiquetas inválidas (0) y atípicos con regla IQR; no se eliminaron valores extremos sin prueba de error.
- `duration` se excluyó por fuga de información: la duración de una llamada no puede usarse para un pronóstico previo a conocerla.
- `pdays=-1` se conserva y significa que no hubo contacto previo.
- El hash **SHA-256** de archivos bruto y limpio queda en `manifest_dataset.json`; el archivo de datos limpios lleva una abreviación del hash.
- Divisiones: 80% entrenamiento / 20% prueba, estratificadas, semilla 42. CV estratificada de 5 pliegues.

## 6. Modelos y ajuste

- **Logística:** `StandardScaler` para numéricas; `OneHotEncoder` para categóricas; búsqueda `C=[0.1,1,10]`, `class_weight=[None,'balanced']` (6 configuraciones).
- **Bosque:** árboles=150, `max_depth=[12,None]`, `min_samples_leaf=[1,5]`, `class_weight=[None,'balanced_subsample']` (8 configuraciones).
- Imputación por mediana/moda dentro de cada pipeline y pliegue. `GridSearchCV(refit='pr_auc')`, `StratifiedKFold(5,shuffle=True,random_state=42)`.
- Selección por PR-AUC promedio **solo de entrenamiento**, evaluación final una sola vez en test.

## 7. Registro MLflow y evidencia

En la ejecución original existen dos corridas finalizadas en `mlruns/` bajo el experimento `Actividad5_BankMarketing_Kaggle`. Incluyen parámetros, métricas de CV y test, registro de modelos, gráficos, tablas y huellas del dataset. `resultados/tablas/mlflow_corridas.csv` resume ambos run_id y `evidencias/mlflow_registros_verificados.png` muestra una **reconstrucción verificable desde archivos del seguimiento**, **no una captura del panel de MLflow**.

**Portabilidad:** las referencias originales a artefactos en `mlruns/*/meta.yaml` contienen `/content/Actividad5/` porque proceden de Colab. Para inspección visual exacta, reproduce el notebook en Colab o ejecuta `python fuentes/preparar_mlflow_local.py` y levanta la UI sobre la copia generada. La copia ajusta rutas de artefactos al directorio local sin modificar el archivo de seguimiento original.

### Cómo subir a GitHub

1. Crea en GitHub un repositorio vacío `actividad5-bank-marketing-mlflow` (preferentemente privado mientras verificas las condiciones de distribución del dataset de Kaggle).
2. El ZIP **ya contiene el historial `.git` inicializado, rama `main` y commit local**. Extrae la carpeta `Actividad5`, abre una terminal dentro de ella y ejecuta:

```bash
git status
git remote add origin https://github.com/TU_USUARIO/actividad5-bank-marketing-mlflow.git
git push -u origin main
```

Si la descompresión o el traslado borró `.git`, puedes reconstruir el repositorio con `git init -b main`, `git add .`, `git commit -m "Entrega Actividad 5"` y después añadir el remoto.

Sustituye `TU_USUARIO` por tu usuario de GitHub. No se ha publicado el repositorio automáticamente.

## 8. Limitaciones y MLOps

Con solo 11.7% de clase positiva y umbral 0.5, los modelos omiten la mayoría de los positivos. Para uso real: validar por fecha y población, definir costos FP/FN y umbrales por objetivo de negocio, calibrar probabilidades, monitorizar drift y fairness, proteger atributos demográficos, automatizar tests y migrar MLflow a un backend SQL para operación colaborativa.

## 9. Atribución y referencias

- Dataset espejo en Kaggle: https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets
- Fuente original UCI: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Documentación MLflow: https://mlflow.org/docs/latest/ml/tracking/
- `scikit-learn` Pipeline y GridSearchCV: https://scikit-learn.org/stable/modules/grid_search.html

La fuente original UCI atribuye el dataset a la campaña bancaria descrita por Moro, Cortez y Rita (2014). Este paquete académico conserva datos tal como fueron descargados; antes de hacer público su contenido revisa las condiciones de redistribución aplicables a esa copia de Kaggle.
