import { Droplets, GitBranch, Database, Brain, Map, BarChart3, Download, Layers } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Hero */}
      <div className="bg-gradient-to-br from-blue-600 to-cyan-600 rounded-2xl p-8 text-white shadow-xl">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-14 h-14 bg-white/20 rounded-2xl flex items-center justify-center backdrop-blur-sm">
            <Droplets className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Water Network Leak Predictor</h1>
            <p className="text-blue-100 text-sm mt-1">v4.0 &mdash; Simulation + Machine Learning + Interactive Map</p>
          </div>
        </div>
        <p className="text-blue-50 leading-relaxed max-w-2xl">
          An interactive dashboard for simulating realistic water distribution networks over
          Sacramento, CA, generating leak events based on Weibull deterioration models, and
          predicting future pipe failures using individual and ensemble ML models. Features
          interactive deck.gl maps, comprehensive analysis, and full data export.
        </p>
      </div>

      {/* How It Works */}
      <div>
        <h2 className="text-lg font-bold text-slate-800 mb-4">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <StepCard
            step={1}
            icon={<Database className="w-5 h-5" />}
            title="Simulate Network"
            description="Generate a realistic pipe network over Sacramento, CA using WNTR hydraulic simulation. Pipes have spatially correlated materials, ages, soil types, and hydraulic properties. The Sacramento River is modeled as an exclusion zone."
          />
          <StepCard
            step={2}
            icon={<BarChart3 className="w-5 h-5" />}
            title="Generate Events"
            description="Leak events are simulated using Weibull failure models with covariate modifiers (soil corrosivity, pressure, repairs, pipe diameter). Seasonal patterns modulate failure rates with winter freeze-thaw peaks."
          />
          <StepCard
            step={3}
            icon={<Brain className="w-5 h-5" />}
            title="Predict & Analyze"
            description="Train individual or ensemble ML models with temporal train/test splits. Explore results via interactive maps, detailed analysis charts, and export data as CSV or full HTML reports."
          />
        </div>
      </div>

      {/* Technical Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <DetailCard title="Simulation Engine">
          <ul className="space-y-2 text-sm text-slate-600">
            <li><strong>WNTR</strong> &mdash; Water Network Tool for Resilience (EPANET-based hydraulic simulation with pressure-driven analysis)</li>
            <li><strong>Weibull Deterioration</strong> &mdash; Material-specific shape (beta) and scale (eta) parameters from water main break rate literature</li>
            <li><strong>Spatial Correlation</strong> &mdash; 5 latitudinal zones with consistent installation eras, materials, and soil types</li>
            <li><strong>Sacramento River</strong> &mdash; Realistic exclusion zone creating natural network partitioning with automatic connectivity repair</li>
            <li><strong>Seasonal Modulation</strong> &mdash; Winter freeze-thaw cycles increase failure rates by up to 40%</li>
            <li><strong>Realistic Topology</strong> &mdash; Irregular grid with removed edges, diagonal connections, arterial mains, and variable block sizes</li>
          </ul>
        </DetailCard>

        <DetailCard title="Machine Learning Pipeline">
          <ul className="space-y-2 text-sm text-slate-600">
            <li><strong>Temporal Splitting</strong> &mdash; Train on historical data, test on future window (no data leakage)</li>
            <li><strong>Feature Engineering</strong> &mdash; Static pipe attributes + historical leak counts, days since last leak, recent leak frequency</li>
            <li><strong>Class Imbalance</strong> &mdash; Handled via scale_pos_weight (XGBoost/LightGBM) or class_weight='balanced' (RF, LR)</li>
            <li><strong>Optimal Threshold</strong> &mdash; F1-maximizing threshold found automatically (not fixed at 0.5)</li>
            <li><strong>Calibration</strong> &mdash; Reliability diagrams show predicted vs actual positive rates</li>
            <li><strong>Honest Metrics</strong> &mdash; All metrics reported as-is, including PR AUC (better for imbalanced data)</li>
          </ul>
        </DetailCard>

        <DetailCard title="Available Models">
          <ul className="space-y-2 text-sm text-slate-600">
            <li className="font-semibold text-slate-500 text-xs uppercase tracking-wider pt-1">Individual</li>
            <li><strong>XGBoost</strong> &mdash; Gradient boosted trees, 200 trees, depth 5, with scale_pos_weight</li>
            <li><strong>LightGBM</strong> &mdash; Fast gradient boosting with is_unbalance flag</li>
            <li><strong>Random Forest</strong> &mdash; 300 decision trees with balanced class weights</li>
            <li><strong>Gradient Boosting</strong> &mdash; Sklearn implementation with sample weighting</li>
            <li><strong>Logistic Regression</strong> &mdash; Linear model, fast and interpretable baseline</li>
            <li className="font-semibold text-slate-500 text-xs uppercase tracking-wider pt-2">Ensemble</li>
            <li><strong>Stacking</strong> &mdash; XGB + LGBM + RF + ExtraTrees with Logistic Regression meta-learner</li>
            <li><strong>Voting</strong> &mdash; Soft voting across XGB + LGBM + RF + GB + MLP</li>
            <li><strong>Blended Boosting</strong> &mdash; Averaged predictions from 2x XGBoost + 2x LightGBM</li>
          </ul>
        </DetailCard>

        <DetailCard title="Technology Stack">
          <ul className="space-y-2 text-sm text-slate-600">
            <li><strong>Backend</strong> &mdash; FastAPI + Python, WNTR, scikit-learn, XGBoost, LightGBM, GeoPandas</li>
            <li><strong>Frontend</strong> &mdash; React 19, TypeScript, TailwindCSS, Recharts, deck.gl, MapLibre GL</li>
            <li><strong>Maps</strong> &mdash; Interactive deck.gl LineLayer &amp; ScatterplotLayer with risk-based coloring and hotspot views</li>
            <li><strong>Export</strong> &mdash; CSV downloads for pipes and events, full HTML analysis reports with embedded charts</li>
          </ul>
        </DetailCard>
      </div>

      {/* Key References */}
      <div className="bg-slate-50 rounded-xl border border-slate-200 p-6">
        <h2 className="text-lg font-bold text-slate-800 mb-3">Key References</h2>
        <ul className="space-y-1.5 text-sm text-slate-600">
          <li>Kleiner, Y. & Rajani, B. (2001). Comprehensive review of structural deterioration of water mains. <em>Urban Water</em>, 3(3), 151-164.</li>
          <li>WNTR: Water Network Tool for Resilience (Sandia National Laboratories)</li>
          <li>Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. <em>KDD 2016</em>.</li>
          <li>Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. <em>NeurIPS 2017</em>.</li>
        </ul>
      </div>
    </div>
  );
}

function StepCard({ step, icon, title, description }: { step: number; icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
      <div className="flex items-center gap-3 mb-3">
        <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-sm font-bold">
          {step}
        </div>
        <div className="text-blue-600">{icon}</div>
        <h3 className="font-semibold text-slate-800">{title}</h3>
      </div>
      <p className="text-sm text-slate-500 leading-relaxed">{description}</p>
    </div>
  );
}

function DetailCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
      <h3 className="font-semibold text-slate-800 mb-3">{title}</h3>
      {children}
    </div>
  );
}
