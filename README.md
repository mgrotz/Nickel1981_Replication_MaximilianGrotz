# Replicating Nickel (1981)

This project simulates dynamic panel data models with fixed effects and estimates the within-transformed regression as described in Nickell (1981).

## 📦 Features

- Simulates:
  - Dynamic panel model with lagged dependent variable
  - Panel model with AR(1) errors

## 📁 Structure

- `explore.py`: Simulation of `y`, `x`, `eps` matrice
- `main.py`: End-to-end pipeline

## ⚙️ Requirements

- `numpy`
- `matplotlib`
- `statsmodels`

Install via:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

```bash
python main.py
```

## 📚 References

- Nickell, S. (1981). *Biases in Dynamic Models with Fixed Effects*. **Econometrica**, 49(6), 1417–1426.  
  [https://www.jstor.org/stable/1911408](https://www.jstor.org/stable/1911408)

  ## 👨‍💻 Author

**Maximilian Grotz**  
Future PhD Student in Economics
Toulouse School of Economics  
📧 [maximiliangrotz@outlook.com](mailto:maximiliangrotz@outlook.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/maximilian-grotz/) | [GitHub](https://github.com/mgrotz)


