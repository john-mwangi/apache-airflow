CREATE TABLE IF NOT EXISTS training_batch (
    patient_id SERIAL PRIMARY KEY,
	mean_radius NUMERIC NOT NULL,
	mean_texture NUMERIC NOT NULL,
	mean_perimeter NUMERIC NOT NULL,
	mean_area NUMERIC NOT NULL,
	mean_smoothness NUMERIC NOT NULL,
	mean_compactness NUMERIC NOT NULL,
	mean_concavity NUMERIC NOT NULL,
	mean_concave_points NUMERIC NOT NULL,
	mean_symmetry NUMERIC NOT NULL,
	mean_fractal_dimension NUMERIC NOT NULL,
	radius_error NUMERIC NOT NULL,
	texture_error NUMERIC NOT NULL,
	perimeter_error NUMERIC NOT NULL,
	area_error NUMERIC NOT NULL,
	smoothness_error NUMERIC NOT NULL,
	compactness_error NUMERIC NOT NULL,
	concavity_error NUMERIC NOT NULL,
	concave_points_error NUMERIC NOT NULL,
	symmetry_error NUMERIC NOT NULL,
	fractal_dimension_error NUMERIC NOT NULL,
	worst_radius NUMERIC NOT NULL,
	worst_texture NUMERIC NOT NULL,
	worst_perimeter NUMERIC NOT NULL,
	worst_area NUMERIC NOT NULL,
	worst_smoothness NUMERIC NOT NULL,
	worst_compactness NUMERIC NOT NULL,
	worst_concavity NUMERIC NOT NULL,
	worst_concave_points NUMERIC NOT NULL,
	worst_symmetry NUMERIC NOT NULL,
	worst_fractal_dimension NUMERIC NOT NULL,
	label NUMERIC NOT NULL
);