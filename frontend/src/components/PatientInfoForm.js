import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { User, Loader2 } from 'lucide-react';

const PatientInfoForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    doctor_name: 'Dr. Emily Johnson',
    consultation_type: 'child_allergy',
    specialty: 'pediatrics',
    age_group: 'child',
    gender: 'both',
    clinic_name: 'Metro Allergy Clinic'
  });

  const consultationTypes = [
    { value: 'child_allergy', label: 'Child Allergy Consultation' },
    { value: 'adult_allergy', label: 'Adult Allergy Consultation' },
    { value: 'child_vaccination', label: 'Child Vaccination' },
    { value: 'general_consultation', label: 'General Consultation' }
  ];

  const specialties = [
    { value: 'pediatrics', label: 'Pediatrics' },
    { value: 'internal_medicine', label: 'Internal Medicine' },
    { value: 'allergy', label: 'Allergy & Immunology' },
    { value: 'general', label: 'General Practice' }
  ];

  const ageGroups = [
    { value: 'child', label: 'Child (0-18 years)' },
    { value: 'adult', label: 'Adult (18+ years)' },
    { value: 'both', label: 'Both' }
  ];

  const genders = [
    { value: 'male', label: 'Male' },
    { value: 'female', label: 'Female' },
    { value: 'both', label: 'Both' }
  ];

  const handleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="card max-w-2xl mx-auto"
    >
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center">
          <User className="w-5 h-5 text-primary-600" />
        </div>
        <h2 className="text-xl font-semibold text-gray-900">Patient Information</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid md:grid-cols-2 gap-6">
          {/* Doctor Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Doctor Name
            </label>
            <input
              type="text"
              value={formData.doctor_name}
              onChange={(e) => handleChange('doctor_name', e.target.value)}
              className="input-field"
              required
            />
          </div>

          {/* Clinic Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Clinic Name
            </label>
            <input
              type="text"
              value={formData.clinic_name}
              onChange={(e) => handleChange('clinic_name', e.target.value)}
              className="input-field"
              required
            />
          </div>
        </div>

        {/* Consultation Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Consultation Type
          </label>
          <select
            value={formData.consultation_type}
            onChange={(e) => handleChange('consultation_type', e.target.value)}
            className="input-field"
            required
          >
            {consultationTypes.map(type => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Specialty */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Medical Specialty
            </label>
            <select
              value={formData.specialty}
              onChange={(e) => handleChange('specialty', e.target.value)}
              className="input-field"
              required
            >
              {specialties.map(specialty => (
                <option key={specialty.value} value={specialty.value}>
                  {specialty.label}
                </option>
              ))}
            </select>
          </div>

          {/* Age Group */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Age Group
            </label>
            <select
              value={formData.age_group}
              onChange={(e) => handleChange('age_group', e.target.value)}
              className="input-field"
              required
            >
              {ageGroups.map(age => (
                <option key={age.value} value={age.value}>
                  {age.label}
                </option>
              ))}
            </select>
          </div>

          {/* Gender */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Gender
            </label>
            <select
              value={formData.gender}
              onChange={(e) => handleChange('gender', e.target.value)}
              className="input-field"
              required
            >
              {genders.map(gender => (
                <option key={gender.value} value={gender.value}>
                  {gender.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="pt-4">
          <button
            type="submit"
            disabled={isLoading}
            className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed py-3"
          >
            {isLoading ? (
              <div className="flex items-center justify-center space-x-2">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Starting Consultation...</span>
              </div>
            ) : (
              'Start Consultation'
            )}
          </button>
        </div>
      </form>
    </motion.div>
  );
};

export default PatientInfoForm; 