# Advanced Foundation: AI Economic Simulation Framework

This project is an implementation of the Foundation framework, a flexible, modular, and composable environment for **modeling socio-economic behaviors and dynamics in a society with both agents and governments**.

It provides a [Gym](https://gym.openai.com/)-style API:
- `reset`: Resets the environment and returns the observation.
- `step`: Moves the environment one step forward, returning the tuple *(observation, reward, done, info)*.

The simulation framework can be used with reinforcement learning to learn optimal economic policies, as detailed in the following papers:

For more information, check out:
- [The AI Economist](https://www.einstein.ai/the-ai-economist)
- [Blog: The AI Economist Moonshot](https://blog.einstein.ai/the-ai-economist-moonshot/)
- [Web demo: AI policy design and COVID-19 case study review by AI Economist](https://einstein.ai/the-ai-economist/ai-policy-foundation-and-covid-case-study)

## Ethics Review and Intended Use
See our [Simulation Card](https://github.com/timipani/advanced-ai-economist/blob/master/Simulation_Card_Foundation_Economic_Simulation_Framework.pdf) for a review of the framework's intended use and ethical review.

## Join our Slack
For extending this framework, discussing machine learning for economics, and collaborating on research projects, [join our Slack channel aieconomist.slack.com](https://aieconomist.slack.com).

## Installation Instructions
You'll need Python 3.7+ installed to get started.

### Pip Installation
You can use the Python package manager: `pip install advanced-ai-economist`

### Source Installation
1. Clone the repository: `git clone www.github.com/timipani/advanced-ai-economist`
2. Create a new conda environment: 
```
conda create --name advanced-ai-economist python=3.7 --yes
conda activate advanced-ai-economist
```
3. Set your PYTHONPATH to include the `advanced-ai-economist` directory: `export PYTHONPATH=<path-to-advanced-ai-economist>:$PYTHONPATH`

## Getting Started
Familiarize with Foundation by trying the tutorials in the 'tutorials' folder.

## Code Structure
The simulation is located in the 'ai_economist/foundation' folder.
- [base](https://www.github.com/timipani/advanced-ai-economist/blob/master/ai_economist/foundation/base): Base classes for defining Agents, Components, and Scenarios.
- [agents](https://www.github.com/timipani/advanced-ai-economist/blob/master/ai_economist/foundation/agents): Agents represent economic actors in the environment.

## Releases and Contributions
Bug reports, pull requests, and other contributions are welcome. See our [contribution guidelines](https://www.github.com/timipani/advanced-ai-economist/blob/master/CONTRIBUTING.md).