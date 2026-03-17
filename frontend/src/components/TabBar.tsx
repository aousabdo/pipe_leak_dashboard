import { BarChart3, Map, FlaskConical, Cpu } from "lucide-react";

const ICONS: Record<string, React.ReactNode> = {
  Overview: <BarChart3 className="w-4 h-4" />,
  Map: <Map className="w-4 h-4" />,
  Analysis: <FlaskConical className="w-4 h-4" />,
  Model: <Cpu className="w-4 h-4" />,
};

interface Props {
  tabs: string[];
  active: string;
  onSelect: (tab: string) => void;
}

export default function TabBar({ tabs, active, onSelect }: Props) {
  return (
    <div className="bg-white border-b border-slate-200 px-6">
      <nav className="flex gap-1">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => onSelect(tab)}
            className={`flex items-center gap-1.5 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              active === tab
                ? "border-blue-600 text-blue-600"
                : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
            }`}
          >
            {ICONS[tab]}
            {tab}
          </button>
        ))}
      </nav>
    </div>
  );
}
