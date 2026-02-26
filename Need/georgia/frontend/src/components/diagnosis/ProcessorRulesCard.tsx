'use client';

import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axiosInstance from '@/lib/axiosInstance';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Loader2, PlusCircle, Edit, Trash2, AlertTriangle, Settings2 } from 'lucide-react';
import { toast } from 'sonner';
import { isAxiosError } from 'axios';

interface ProcessorRule {
  id?: string;
  ruleName: string;
  description?: string;
  documentTypeLabel: string;
  // classifierProcessorId: string; // Removed
  targetParserProcessorId: string;
  isEnabled: boolean;
  priority: number;
  createdAt?: string;
  updatedAt?: string;
}

const initialRuleFormData: ProcessorRule = {
  ruleName: '',
  documentTypeLabel: '',
  // classifierProcessorId: '', // Removed
  targetParserProcessorId: '',
  isEnabled: true,
  priority: 10,
  description: '',
};

export function ProcessorRulesCard() {
  const queryClient = useQueryClient();
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [currentRule, setCurrentRule] = useState<ProcessorRule | null>(null);
  const [formData, setFormData] = useState<ProcessorRule>(initialRuleFormData);
  // For delete confirmation dialog
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [ruleToDelete, setRuleToDelete] = useState<ProcessorRule | null>(null);


  const { data: rules, isLoading: isLoadingRules, error: rulesError } = useQuery<ProcessorRule[]>({
    queryKey: ['processorRules'],
    queryFn: async () => {
      // Fetch ALL rules for the admin management card, regardless of enabled status
      const response = await axiosInstance.get('/processor-rules?enabled_only=false');
      return response.data;
    },
  });

  useEffect(() => {
    if (currentRule) {
      setFormData(currentRule);
    } else {
      setFormData(initialRuleFormData);
    }
  }, [currentRule]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    // Check if the event target is an HTMLInputElement for 'type' and 'checked' properties
    if (e.target instanceof HTMLInputElement) {
      const { type, checked } = e.target;
      if (type === 'checkbox') { // This case is not used currently but good for future proofing
        setFormData(prev => ({ ...prev, [name]: checked }));
      } else if (type === 'number') {
        setFormData(prev => ({ ...prev, [name]: parseInt(value, 10) || 0 }));
      } else {
        setFormData(prev => ({ ...prev, [name]: value }));
      }
    } else { // For HTMLTextAreaElement (if any, currently not used for named inputs)
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };
  
  const handleSwitchChange = (checked: boolean) => {
    setFormData(prev => ({ ...prev, isEnabled: checked }));
  };

  const mutation = useMutation<ProcessorRule, Error, ProcessorRule>({
    mutationFn: async (ruleData) => {
      if (currentRule?.id) { // Editing existing rule
        const response = await axiosInstance.put(`/processor-rules/${currentRule.id}`, ruleData);
        return response.data;
      } else { // Creating new rule
        const response = await axiosInstance.post('/processor-rules', ruleData);
        return response.data;
      }
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['processorRules'] });
      toast.success(currentRule?.id ? 'Processor rule updated!' : 'Processor rule created!');
      setIsFormOpen(false);
      setCurrentRule(null);
    },
    onError: (error) => {
      let errorMsg = 'Failed to save processor rule.';
      if (isAxiosError(error) && error.response?.data?.error) {
        errorMsg = error.response.data.error;
      } else if (error instanceof Error) {
        errorMsg = error.message;
      }
      toast.error(errorMsg);
    },
  });

  const deleteMutation = useMutation<void, Error, string>({
    mutationFn: async (ruleId: string) => {
      await axiosInstance.delete(`/processor-rules/${ruleId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['processorRules'] });
      toast.success('Processor rule deleted!');
    },
    onError: (error) => {
      let errorMsg = 'Failed to delete processor rule.';
       if (isAxiosError(error) && error.response?.data?.error) {
        errorMsg = error.response.data.error;
      } else if (error instanceof Error) {
        errorMsg = error.message;
      }
      toast.error(errorMsg);
    },
  });


  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    mutation.mutate(formData);
  };

  const openAddModal = () => {
    setCurrentRule(null);
    setFormData(initialRuleFormData);
    setIsFormOpen(true);
  };

  const openEditModal = (rule: ProcessorRule) => {
    setCurrentRule(rule);
    setFormData(rule);
    setIsFormOpen(true);
  };
  
  const handleDelete = (ruleId: string | undefined) => {
    if (!ruleId) return;
    const rule = rules?.find(r => r.id === ruleId);
    if (rule) {
      setRuleToDelete(rule);
      setIsDeleteDialogOpen(true);
    }
  };

  const confirmDelete = () => {
    if (ruleToDelete?.id) {
      deleteMutation.mutate(ruleToDelete.id);
    }
    setIsDeleteDialogOpen(false);
    setRuleToDelete(null);
  };

  const toggleEnableMutation = useMutation<ProcessorRule, Error, { id: string, isEnabled: boolean }>({
    mutationFn: async ({ id, isEnabled }) => {
      // We only need to send the isEnabled field for this partial update
      const response = await axiosInstance.put(`/processor-rules/${id}`, { isEnabled });
      return response.data; // Assuming backend returns the updated rule or success message
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['processorRules'] });
      toast.success('Rule status updated!');
    },
    onError: (error) => {
      let errorMsg = 'Failed to update rule status.';
      if (isAxiosError(error) && error.response?.data?.error) {
        errorMsg = error.response.data.error;
      } else if (error instanceof Error) {
        errorMsg = error.message;
      }
      toast.error(errorMsg);
      // Optionally refetch to revert optimistic update if any, or rely on query invalidation
      queryClient.invalidateQueries({ queryKey: ['processorRules'] });
    },
  });

  const handleToggleEnable = (ruleId: string | undefined, currentIsEnabled: boolean) => {
    if (!ruleId) return;
    toggleEnableMutation.mutate({ id: ruleId, isEnabled: !currentIsEnabled });
  };


  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
                <Settings2 className="h-6 w-6" />
                <CardTitle>Document Processor Routing Rules</CardTitle>
            </div>
            <Button onClick={openAddModal} size="sm">
                <PlusCircle className="mr-2 h-4 w-4" /> Add New Rule
            </Button>
        </div>
        <CardDescription>
          Manage rules for classifying documents and selecting specific Document AI processors.
          Lower priority numbers are evaluated first.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoadingRules && (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            <span className="ml-2 text-muted-foreground">Loading rules...</span>
          </div>
        )}
        {rulesError && (
          <div className="text-red-600 flex items-center">
            <AlertTriangle className="mr-2 h-5 w-5" /> Error loading rules: {rulesError.message}
          </div>
        )}
        {!isLoadingRules && !rulesError && rules && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Rule Name</TableHead>
                <TableHead>Type Label</TableHead>
                {/* <TableHead>Classifier ID</TableHead> // Removed */}
                <TableHead>Parser ID</TableHead>
                <TableHead className="text-center">Priority</TableHead>
                <TableHead className="text-center">Enabled</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rules.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center text-muted-foreground py-8"> {/* Adjusted colSpan */}
                    No processor routing rules defined yet.
                  </TableCell>
                </TableRow>
              )}
              {rules.map((rule) => (
                <TableRow key={rule.id}>
                  <TableCell className="font-medium">{rule.ruleName}</TableCell>
                  <TableCell>{rule.documentTypeLabel}</TableCell>
                  {/* <TableCell className="text-xs truncate max-w-[150px]" title={rule.classifierProcessorId}>{rule.classifierProcessorId}</TableCell> // Removed */}
                  <TableCell className="text-xs truncate max-w-[150px]" title={rule.targetParserProcessorId}>{rule.targetParserProcessorId}</TableCell>
                  <TableCell className="text-center">{rule.priority}</TableCell>
                  <TableCell className="text-center">
                    <Switch
                      checked={rule.isEnabled}
                      onCheckedChange={() => handleToggleEnable(rule.id, rule.isEnabled)}
                      disabled={toggleEnableMutation.isPending && toggleEnableMutation.variables?.id === rule.id}
                      aria-label={rule.isEnabled ? "Disable rule" : "Enable rule"}
                    />
                  </TableCell>
                  <TableCell className="text-right space-x-2">
                    <Button variant="outline" size="icon" onClick={() => openEditModal(rule)} disabled={toggleEnableMutation.isPending}>
                      <Edit className="h-4 w-4" />
                    </Button>
                    <Button variant="destructive" size="icon" onClick={() => handleDelete(rule.id)}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>

      <Dialog open={isFormOpen} onOpenChange={setIsFormOpen}>
        <DialogContent className="sm:max-w-[625px]">
          <DialogHeader>
            <DialogTitle>{currentRule?.id ? 'Edit' : 'Add New'} Processor Rule</DialogTitle>
            <DialogDescription>
              {currentRule?.id ? 'Update the details of the processor routing rule.' : 'Define a new rule for document classification and processor selection.'}
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleSubmit} className="space-y-4 py-4">
            <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                    <Label htmlFor="ruleName">Rule Name</Label>
                    <Input id="ruleName" name="ruleName" value={formData.ruleName} onChange={handleInputChange} required />
                </div>
                <div className="space-y-2">
                    <Label htmlFor="documentTypeLabel">Document Type Label</Label>
                    <Input id="documentTypeLabel" name="documentTypeLabel" value={formData.documentTypeLabel} onChange={handleInputChange} placeholder="e.g., INVOICE_VENDOR_X" required />
                </div>
            </div>
            <div className="space-y-2">
                <Label htmlFor="description">Description (Optional)</Label>
                <Input id="description" name="description" value={formData.description || ''} onChange={handleInputChange} />
            </div>
            {/* Classifier Processor ID field removed from form */}
            {/* 
            <div className="space-y-2">
              <Label htmlFor="classifierProcessorId">Classifier Processor ID</Label>
              <Input id="classifierProcessorId" name="classifierProcessorId" value={formData.classifierProcessorId} onChange={handleInputChange} placeholder="projects/.../processors/your-classifier-id" required />
              <p className="text-xs text-muted-foreground">Full Document AI processor ID for the classifier.</p>
            </div>
            */}
            <div className="space-y-2">
              <Label htmlFor="targetParserProcessorId">Target Parser Processor ID (Short ID)</Label>
              <Input id="targetParserProcessorId" name="targetParserProcessorId" value={formData.targetParserProcessorId} onChange={handleInputChange} placeholder="e.g., e747161e81eca740" required />
               <p className="text-xs text-muted-foreground">Enter the short ID of the Document AI parser (e.g., from processor details page in GCP console).</p>
            </div>
            <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                    <Label htmlFor="priority">Priority</Label>
                    <Input id="priority" name="priority" type="number" value={formData.priority} onChange={handleInputChange} required />
                    <p className="text-xs text-muted-foreground">Lower numbers run first.</p>
                </div>
                <div className="space-y-2 flex flex-col pt-2">
                    <Label htmlFor="isEnabled" className="mb-2">Enabled</Label>
                    <Switch id="isEnabled" name="isEnabled" checked={formData.isEnabled} onCheckedChange={handleSwitchChange} />
                </div>
            </div>
            <DialogFooter>
              <DialogClose asChild>
                <Button type="button" variant="outline">Cancel</Button>
              </DialogClose>
              <Button type="submit" disabled={mutation.isPending}>
                {mutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {currentRule?.id ? 'Save Changes' : 'Create Rule'}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Custom Delete Confirmation Dialog */}
      {ruleToDelete && (
        <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Confirm Deletion</DialogTitle>
              <DialogDescription>
                Are you sure you want to delete the rule "<strong>{ruleToDelete.ruleName}</strong>"? This action cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsDeleteDialogOpen(false)} disabled={deleteMutation.isPending}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={confirmDelete} disabled={deleteMutation.isPending}>
                {deleteMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </Card>
  );
}
