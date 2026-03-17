import { useState, useEffect, useCallback } from "react";
import { api } from "./api";
import { useAsync } from "./hooks/useApi";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import TabBar from "./components/TabBar";
import RerunBanner from "./components/RerunBanner";
import LoadingOverlay from "./components/LoadingOverlay";
import OverviewPage from "./pages/OverviewPage";
import MapPage from "./pages/MapPage";
import AnalysisPage from "./pages/AnalysisPage";
import ModelPage from "./pages/ModelPage";
import AboutPage from "./pages/AboutPage";

const TABS = ["Overview", "Map", "Analysis", "Model", "About"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Overview");
  const [params, setParams] = useState({ num_pipes: 1000, sim_years: 5, seed: 42 });
  const [appliedParams, setAppliedParams] = useState<typeof params | null>(null);
  const [dirty, setDirty] = useState(false);
  const [modelType, setModelType] = useState("xgboost");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const status = useAsync<any>();
  const overview = useAsync<any>();
  const pipes = useAsync<any>();
  const analysis = useAsync<any>();
  const model = useAsync<any>();
  const filters = useAsync<any>();

  const [simulating, setSimulating] = useState(false);
  const [training, setTraining] = useState(false);

  // Check if params changed from last applied
  useEffect(() => {
    if (!appliedParams) return;
    const changed =
      params.num_pipes !== appliedParams.num_pipes ||
      params.sim_years !== appliedParams.sim_years ||
      params.seed !== appliedParams.seed;
    setDirty(changed);
  }, [params, appliedParams]);

  const loadAllData = useCallback(async () => {
    await Promise.all([
      overview.run(() => api.getOverview()),
      pipes.run(() => api.getPipes()),
      analysis.run(() => api.getAnalysis()),
      filters.run(() => api.getFilters()),
    ]);
    try {
      await model.run(() => api.getModel());
    } catch {
      // Model may not exist yet
    }
  }, []);

  // Initial load
  useEffect(() => {
    (async () => {
      try {
        const s = await status.run(() => api.getStatus());
        if (s.has_data) {
          setAppliedParams(s.sim_params || params);
          await loadAllData();
        }
      } catch {
        // API not ready yet
      }
    })();
  }, []);

  const handleSimulate = useCallback(async () => {
    setSimulating(true);
    try {
      await api.simulate(params);
      setAppliedParams({ ...params });
      setDirty(false);
      await loadAllData();
    } catch (e: any) {
      alert("Simulation failed: " + e.message);
    } finally {
      setSimulating(false);
    }
  }, [params, loadAllData]);

  const handleTrain = useCallback(async () => {
    setTraining(true);
    try {
      await api.train(modelType);
      await Promise.all([
        model.run(() => api.getModel()),
        overview.run(() => api.getOverview()),
        pipes.run(() => api.getPipes()),
      ]);
    } catch (e: any) {
      alert("Training failed: " + e.message);
    } finally {
      setTraining(false);
    }
  }, [modelType]);

  const hasData = overview.data !== null;
  const hasModel = model.data !== null;

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      {/* Sidebar */}
      <Sidebar
        params={params}
        onParamsChange={setParams}
        onSimulate={handleSimulate}
        onTrain={handleTrain}
        simulating={simulating}
        training={training}
        hasData={hasData}
        hasModel={hasModel}
        dirty={dirty}
        modelType={modelType}
        onModelTypeChange={setModelType}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header
          numPipes={overview.data?.kpis?.total_pipes}
          numEvents={overview.data?.kpis?.total_events}
          hasModel={hasModel}
          modelType={model.data?.model_type}
        />

        {dirty && hasData && (
          <RerunBanner onRerun={handleSimulate} simulating={simulating} />
        )}

        <TabBar tabs={TABS as unknown as string[]} active={tab} onSelect={(t) => setTab(t as Tab)} />

        <main className="flex-1 overflow-y-auto p-6">
          {!hasData && !simulating ? (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <div className="text-6xl mb-4">💧</div>
                <h2 className="text-2xl font-bold text-slate-700 mb-2">
                  Welcome to Water Network Leak Predictor
                </h2>
                <p className="text-slate-500 mb-6 max-w-md">
                  Configure your simulation parameters in the sidebar, then click
                  <strong> Generate Network</strong> to get started.
                </p>
              </div>
            </div>
          ) : (
            <>
              {tab === "Overview" && <OverviewPage data={overview.data} loading={overview.loading} />}
              {tab === "Map" && <MapPage data={pipes.data} loading={pipes.loading} />}
              {tab === "Analysis" && <AnalysisPage data={analysis.data} loading={analysis.loading} />}
              {tab === "Model" && <ModelPage data={model.data} loading={model.loading} hasModel={hasModel} onTrain={handleTrain} training={training} />}
              {tab === "About" && <AboutPage />}
            </>
          )}
        </main>
      </div>

      {(simulating || training) && (
        <LoadingOverlay message={simulating ? "Generating pipe network & simulating leaks..." : `Training ${modelType.replace('_', ' ')} model...`} />
      )}
    </div>
  );
}
