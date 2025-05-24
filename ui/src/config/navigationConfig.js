import DashboardIcon from '@mui/icons-material/Dashboard';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import SettingsIcon from '@mui/icons-material/Settings';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';

const navigationConfig = [
  {
    label: 'Dashboard',
    path: '/dashboard',
    icon: DashboardIcon,
  },
  {
    label: 'Trading',
    icon: ShowChartIcon,
    subItems: [
      { label: 'Terminal', path: '/terminal' },
      { label: 'Portfolio', path: '/portfolio' },
    ],
  },
  {
    label: 'Analytics',
    path: '/analytics',
    icon: AutoGraphIcon,
  },
  {
    label: 'Settings',
    path: '/settings',
    icon: SettingsIcon,
  },
];

export default navigationConfig;
