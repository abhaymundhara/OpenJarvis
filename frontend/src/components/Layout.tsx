import { Outlet } from 'react-router';
import { Sidebar } from './Sidebar/Sidebar';
import { useAppStore } from '../lib/store';

export function Layout() {
  const sidebarOpen = useAppStore((s) => s.sidebarOpen);

  return (
    <div className="flex h-full w-full overflow-hidden">
      <Sidebar />
      {/* Overlay for mobile when sidebar is open */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/40 md:hidden"
          onClick={() => useAppStore.getState().setSidebarOpen(false)}
        />
      )}
      <main className="flex-1 flex flex-col min-w-0 h-full" style={{ background: 'var(--color-bg)' }}>
        <Outlet />
      </main>
    </div>
  );
}
