interface Props {
  message: string;
}

export default function LoadingOverlay({ message }: Props) {
  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-2xl px-8 py-6 flex flex-col items-center gap-4 max-w-sm">
        <div className="relative">
          <div className="w-16 h-16 rounded-full border-4 border-slate-200 border-t-blue-600 animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center text-2xl">💧</div>
        </div>
        <p className="text-sm font-medium text-slate-600 text-center">{message}</p>
        <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
          <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full animate-[loading_2s_ease-in-out_infinite]"
               style={{ width: "70%", animation: "loading 2s ease-in-out infinite" }} />
        </div>
      </div>
      <style>{`
        @keyframes loading {
          0% { width: 10%; margin-left: 0%; }
          50% { width: 60%; margin-left: 20%; }
          100% { width: 10%; margin-left: 90%; }
        }
      `}</style>
    </div>
  );
}
