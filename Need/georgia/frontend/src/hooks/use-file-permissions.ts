import { useAuth } from "@/context/AuthContext";

export const useFilePermissions = () => {
    const { user } = useAuth();

    const isSuperAdmin = user?.role === "superadmin";
    const permissions = user?.file_management_permissions;
    const adminPermissions = user?.admin_panel_access_permission;

    return {
        // File Management Permissions
        canRenameSource: isSuperAdmin || permissions?.can_rename_source === true,
        canDeleteSource: isSuperAdmin || permissions?.can_delete_source === true,
        canUpload: isSuperAdmin || permissions?.can_upload === true,
        canCreateRootFolderSource: isSuperAdmin || permissions?.can_create_root_folder_source === true,
        canCreateFolderSource: isSuperAdmin || permissions?.can_create_folder_source === true,
        canDeleteDestination: isSuperAdmin || permissions?.can_delete_destination === true,
        canCreateRootFolderDestination: isSuperAdmin || permissions?.can_create_root_folder_destination === true,
        canCreateFolderDestination: isSuperAdmin || permissions?.can_create_folder_destination === true,
        canTransfer: isSuperAdmin || permissions?.can_transfer === true,
        canViewAllTransferHistory: isSuperAdmin || permissions?.can_view_all_transfer_history === true,

        // Admin Panel Permissions
        canAccessAdminPanel: isSuperAdmin || (
            adminPermissions?.can_access_regions === true ||
            adminPermissions?.can_manage_roles === true ||
            adminPermissions?.can_manage_users === true ||
            adminPermissions?.can_update_settings === true
        ),
        canAccessRegions: isSuperAdmin || adminPermissions?.can_access_regions === true,
        canManageRoles: isSuperAdmin || adminPermissions?.can_manage_roles === true,
        canManageUsers: isSuperAdmin || adminPermissions?.can_manage_users === true,
        canUpdateSettings: isSuperAdmin || adminPermissions?.can_update_settings === true,

        isSuperAdmin,
    };
};
