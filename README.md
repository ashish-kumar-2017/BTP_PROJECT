# 📡 TCP for Network Slicing using Reinforcement Learning

This project leverages **Reinforcement Learning (RL)** to optimize the **Transmission Control Protocol (TCP)** for next-generation networks. It focuses on addressing the challenges of delay-sensitive applications like **virtual reality (VR)**, **remote surgeries**, and **autonomous vehicles** by dynamically managing network resources.

---

## 🚀 Features

- **Reinforcement Learning Framework**:
  - 📊 **State Space**: Captures real-time data rates of three hosts.
  - 🎛 **Action Space**: Dynamically adjusts host data rates.
  - 🏆 **Reward Function**: Ensures delay-bound compliance, maximizes throughput, and promotes fairness.

- **Integration with OMNeT++**:
  - 🌐 Simulates real-time SDN environments.
  - 🔄 Enables dynamic adjustments for multi-host traffic.
  - 📈 Visualizes and analyzes network performance.

- **Performance Optimization**:
  - ⏱ Reduces network delays.
  - 📡 Improves throughput efficiency.
  - ⚖ Ensures resource fairness and compliance with constraints.

---

## 🛠 Setup Instructions

### 1️⃣ System Requirements
- **Languages & Tools**:
  - Python 3.x
  - OMNeT++ 6.0.3 or later
  - MATLAB (optional for advanced modeling)
- **Python Libraries**:
  - `random`, `itertools`, `pickle`

## 2️⃣ Installation
**Clone the repository and set up the environment**:
- bash 
  - git clone https://github.com/ashish-kumar-2017/BTPPROJECT.git
  - cd BTPPROJECT
  - pip install -r requirements.txt
---

## 📸 **Preview-Two Examples**
<p align="center">
  <img src="images/picture1.png" alt="network" width="530"/>
  <img src="images/image2.png" alt="real_life_app" width="460"/>
</p>

## 📊 Results
- Highlights
  - ✅ Stable Network Configuration: Achieved optimal data rates for multi-host scenarios.
  - 🚀 Performance Boost: Significant improvements in:
    - Total throughput.
    - Average delay compliance.
    - Fair resource allocation.
Key Metrics
Metric	Outcome
Throughput	Maximized
Delay Compliance	Maintained consistently
Fairness	Ensured among all hosts

---
## 📊 Key Metrics

| **Metric**                | **Description**                                    | **Outcome**                                    |
|---------------------------|----------------------------------------------------|-----------------------------------------------|
| **Throughput**            | Total data transmitted across the network.         | Maximized for multi-host scenarios.           |
| **Delay Compliance**      | Adherence to strict delay bounds.                  | Maintained consistently across episodes.      |
| **Fairness**              | Balanced resource allocation among hosts.          | Ensured for all traffic conditions.           |
| **Network Stability**     | Stable data rates and reward convergence.          | Achieved after training with RL algorithms.   |
| **Scalability**           | Ability to handle dynamic traffic scenarios.       | Demonstrated in small-scale networks.         |

---

### Example Performance (Episode Progression)
| **Episode** | **Initial Data Rates**  | **Final Data Rates**  | **Reward**  |
|-------------|-------------------------|-----------------------|-------------|
| 6           | `[50, 50, 128]`         | `[0, 0, 0]`           | `-200.0`    |
| 13          | `[50, 50, 128]`         | `[0, 0, 1]`           | `608.0`     |
| 22          | `[60, 70, 40]`          | `[42, 33, 5]`         | `3163.0`    |
| 30+         | `[37, 24, 11]`          | `[43, 34, 8]`         | `9606.0`    |

> **Note**: These metrics demonstrate the effectiveness of the RL model in stabilizing network performance and achieving high rewards through optimal policy training.
---

## 🛑 Challenges & 🔮 Future Directions

### Challenges
- ⚡ **Slow Convergence**: Training required many episodes in high-delay scenarios, leading to prolonged experimentation.
- 🤖 **Exploration-Exploitation Trade-off**: Balancing between exploring new strategies and exploiting known ones posed a challenge.
- 🔗 **Scalability**: Current implementation was tested only on small-scale networks, limiting its applicability to larger systems.

### Future Directions
- 📚 **Explore Advanced RL Algorithms**:
  - Implement **Deep Q-Learning** for better generalization and handling of complex network states.
  - Investigate **Multi-Agent RL** to coordinate multiple RL agents in larger networks.
- 🌍 **Enhance Scalability**:
  - Adapt the model for real-world scenarios with higher traffic loads and larger host configurations.
- 🛠 **Extend OMNeT++ Integration**:
  - Enable adaptive management of multi-host scenarios to handle fluctuating traffic conditions dynamically.

---

## 📚 References

1. ["TCP Slice: A semi-distributed TCP algorithm for Delay-constrained Applications"](http://arxiv.org/abs/2312.01869v1)
2. [OpenAI Gym Documentation](https://www.gymlibrary.dev/)

---

 ## 🙏 Acknowledgments

This project was completed under the guidance of **Prof. Dibbendu Roy**. 

I would like to extend my sincere gratitude to:
- **Prof. Dibbendu Roy** for their invaluable guidance, insights, and mentorship throughout this project.

This journey has been an incredible learning experience, significantly enhancing my skills and understanding of **Reinforcement Learning**, **network optimization**, and **simulation platforms**.

---

**Ashish Kumar**  
IIT Indore

 ---

## 📨 Contact

For queries, suggestions, or collaborations, feel free to reach out:

- 📂 **GitHub Repository**: [https://github.com/ashish-kumar-2017/BTPPROJECT](https://github.com/ashish-kumar-2017/BTPPROJECT)





