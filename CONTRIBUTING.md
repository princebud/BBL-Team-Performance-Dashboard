# Contributing

- Keep the demo dataset clearly labelled as synthetic.
- Do not commit raw Cricsheet archives or real-data CSVs unless the data licence permits it.
- Run the demo processing command before submitting changes:
  `python data_processing.py demo_data/demo_ball_by_ball.csv demo_data/demo_matches.csv --out output`
- Check that the dashboard and notebook use the shared metrics from `data_processing.py`.
- Document metric definitions and caveats when changing calculations.
